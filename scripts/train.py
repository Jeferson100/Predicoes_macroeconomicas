"""Treina, avalia e publica UMA variavel macro.

Uso:
    python -m scripts.train --variable selic

Regras:
- baselines (random walk, seasonal naive, media, drift) sao sempre avaliados;
- o campeao e escolhido por MASE em origens moveis, nunca no holdout;
- se nenhum modelo bate o random walk, o random walk e o campeao;
- o holdout e usado uma unica vez, para reportar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from economic_brazil.forecasting import backtest, baselines, conformal
from economic_brazil.forecasting.features import montar_dataset, montar_linha_futura
from economic_brazil.forecasting.target import AlvoAncorado

RAIZ = Path(__file__).resolve().parents[1]
ARTEFATOS = RAIZ / "artefatos"
VARIAVEIS = ["selic", "ipca", "pib", "taxa_desocupacao", "producao_industrial"]


def criar_modelo(nome: str):
    """Candidatos deliberadamente simples e regularizados."""
    if nome == "ridge":
        return AlvoAncorado(make_pipeline(StandardScaler(), Ridge(alpha=1.0)))

    if nome == "elasticnet":
        return AlvoAncorado(
            make_pipeline(
                StandardScaler(),
                ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50_000, random_state=0),
            )
        )

    if nome == "gradient_boosting":
        return AlvoAncorado(
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                subsample=0.8,
                random_state=0,
            )
        )

    raise ValueError(f"modelo desconhecido: {nome!r}")


def carregar_config(variavel: str) -> dict:
    caminho = RAIZ / "configs" / f"{variavel}.yaml"
    return yaml.safe_load(caminho.read_text(encoding="utf-8"))


def preparar_serie(dados: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.DataFrame | None]:
    frequencia = "QS" if cfg["frequencia"] == "Q" else "MS"
    serie = dados[cfg["alvo"]].astype(float).resample(frequencia).last()
    serie = serie.loc[cfg["data_inicio"] :].dropna()

    colunas = [c for c in cfg.get("exogenas", []) if c in dados.columns]
    exogenas = dados[colunas].astype(float).resample(frequencia).mean()
    exogenas = exogenas.reindex(serie.index) if colunas else None
    return serie, exogenas


def _indice_futuro(serie: pd.Series, cfg: dict) -> pd.Timestamp:
    meses = 3 if cfg["frequencia"] == "Q" else 1
    return serie.index[-1] + pd.DateOffset(months=meses * int(cfg["horizonte"]))


def _previsao_baseline(nome: str, alvo: np.ndarray, cfg: dict) -> float:
    funcao = baselines.BASELINES[nome]
    extra = {"season": int(cfg["season"])} if nome == "seasonal_naive" else {}
    previsao = funcao(alvo, int(cfg["horizonte"]), **extra)
    return float(np.asarray(previsao, dtype=float).ravel()[0])


def avaliar(variavel: str) -> dict:
    cfg = carregar_config(variavel)
    if int(cfg["horizonte"]) != 1:
        raise ValueError("esta versao avalia apenas horizonte=1")

    dados = pd.read_pickle(RAIZ / "dados" / "economic_data_brazil.pkl")
    serie, exogenas = preparar_serie(dados, cfg)

    X, y = montar_dataset(
        serie,
        lags=tuple(cfg["lags"]),
        exogenas=exogenas,
        sazonalidade=bool(cfg["sazonalidade"]),
    )
    if len(y) < 60:
        raise ValueError(f"{variavel}: apenas {len(y)} observacoes apos as features")

    season = int(cfg["season"])
    n_origens = int(cfg["n_origens"])
    alvo = y.to_numpy(dtype=float)
    escala_insample = alvo[:-n_origens]

    backtests: dict[str, pd.DataFrame] = {}
    placar: dict[str, dict] = {}

    for nome in cfg["baselines"]:
        backtests[nome] = backtest.backtest_baseline(
            baselines.BASELINES[nome],
            alvo,
            n_origens=n_origens,
            horizonte=1,
            **({"season": season} if nome == "seasonal_naive" else {}),
        )
        placar[nome] = (
            backtest.resumo(backtests[nome], escala_insample, 1, season_sazonal=season)
            .loc["geral"]
            .to_dict()
        )

    for nome in cfg["candidatos"]:
        backtests[nome] = backtest.backtest_supervisionado(
            lambda n=nome: criar_modelo(n), X, y, n_origens=n_origens
        )
        placar[nome] = (
            backtest.resumo(backtests[nome], escala_insample, 1, season_sazonal=season)
            .loc["geral"]
            .to_dict()
        )

    campeao = min(placar, key=lambda nome: placar[nome]["MASE"])

    if campeao in baselines.BASELINES:
        previsao = _previsao_baseline(campeao, alvo, cfg)
        data_previsao = _indice_futuro(serie, cfg)
    else:
        modelo = criar_modelo(campeao)
        modelo.fit(X, y)
        linha_futura = montar_linha_futura(
            serie,
            lags=tuple(cfg["lags"]),
            exogenas=exogenas,
            sazonalidade=bool(cfg["sazonalidade"]),
        ).reindex(columns=X.columns)
        previsao = float(np.asarray(modelo.predict(linha_futura), dtype=float).ravel()[0])
        data_previsao = linha_futura.index[0]

    calibracao = backtest.erros_absolutos(backtests[campeao])[-int(cfg["n_calib"]) :]
    intervalos = {
        nivel: [round(lower, 4), round(upper, 4)]
        for nivel, (lower, upper) in conformal.intervalos(
            previsao, calibracao, tuple(cfg["alphas"])
        ).items()
    }
    cobertura = {
        f"{round((1 - alpha) * 100)}%": conformal.cobertura_empirica(
            backtests[campeao], alpha, int(cfg["n_calib"])
        )
        for alpha in cfg["alphas"]
    }

    resultado = {
        "variavel": variavel,
        "nome": cfg["nome"],
        "unidade": cfg["unidade"],
        "frequencia": cfg["frequencia"],
        "ancora": cfg["ancora"],
        "horizonte": 1,
        "n_observacoes": int(len(y)),
        "ultima_observacao": str(y.index[-1].date()),
        "ultimo_valor": float(alvo[-1]),
        "data_previsao": str(pd.Timestamp(data_previsao).date()),
        "previsao": round(previsao, 4),
        "intervalos": intervalos,
        "cobertura_empirica": cobertura,
        "placar_mase": {nome: round(v["MASE"], 4) for nome, v in placar.items()},
        "placar_mae": {nome: round(v["MAE"], 4) for nome, v in placar.items()},
        "campeao": campeao,
        "mase_campeao": round(placar[campeao]["MASE"], 4),
        "bate_random_walk": bool(placar[campeao]["MASE"] < 1.0 and campeao != "random_walk"),
        "limites_plausiveis": cfg["limites_plausiveis"],
    }

    print(
        f"\n{cfg['nome']} | ultima obs {resultado['ultima_observacao']} = "
        f"{resultado['ultimo_valor']:.3f} | previsao {resultado['data_previsao']} = "
        f"{previsao:.3f}"
    )
    print(pd.DataFrame(placar).T[["MAE", "RMSE", "vies", "MASE"]].round(3).to_string())
    print(f"campeao: {campeao} (MASE {resultado['mase_campeao']})")

    return resultado


def main(variavel: str) -> int:
    resultado = avaliar(variavel)
    ARTEFATOS.mkdir(exist_ok=True)
    destino = ARTEFATOS / f"metrics-{variavel}.json"
    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nmetricas salvas em {destino}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", required=True, choices=VARIAVEIS)
    raise SystemExit(main(parser.parse_args().variable))
