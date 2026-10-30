"""Backtest de origem movel, MASE e Diebold-Mariano.

Regra do pacote: o conjunto de teste e usado uma unica vez, no fim. Toda
comparacao entre modelos acontece aqui, em origens moveis.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

COLUNAS_RESUMO = ["MAE", "RMSE", "vies", "MASE"]


def mase(y_true, y_pred, y_insample, season: int = 1) -> float:
    """Erro absoluto medio dividido pelo erro absoluto medio do naive.

    MASE < 1 significa que o modelo bate o passeio aleatorio. A escala vem
    de `y_insample`, nunca do conjunto de teste.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_insample = np.asarray(y_insample, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true e y_pred com formatos diferentes: {y_true.shape} vs {y_pred.shape}"
        )
    if len(y_insample) <= season:
        return float("inf")

    numerador = float(np.mean(np.abs(y_true - y_pred)))
    denominador = float(np.mean(np.abs(np.diff(y_insample, n=season))))
    return numerador / denominador if denominador > 0 else float("inf")


def _escala_naive(y_insample, season: int = 1) -> float:
    y_insample = np.asarray(y_insample, dtype=float).ravel()
    if len(y_insample) <= season:
        return float("nan")
    return float(np.mean(np.abs(np.diff(y_insample, n=season))))


def _limitar_origens(n: int, n_origens: int, minimo_treino: int = 30) -> int:
    """Mantem o mesmo numero de origens para todos os modelos, para a comparacao
    entre eles ser justa."""
    return max(1, min(n_origens, n - minimo_treino))


def backtest_baseline(
    prever: Callable[..., np.ndarray],
    y,
    n_origens: int = 24,
    horizonte: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """Origem movel para um baseline. `prever(y_insample, h, **kwargs)` devolve h valores."""
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    n_origens = _limitar_origens(n, n_origens)

    linhas = []
    for origem in range(n - n_origens - horizonte, n - horizonte):
        y_insample = y[: origem + 1]
        previsoes = np.asarray(prever(y_insample, horizonte, **kwargs), dtype=float).ravel()
        for passo in range(horizonte):
            linhas.append(
                {
                    "origem": origem,
                    "passo": passo + 1,
                    "real": y[origem + 1 + passo],
                    "previsto": previsoes[passo],
                }
            )
    return pd.DataFrame(linhas)


def backtest_supervisionado(
    criar_modelo: Callable[[], object],
    X: pd.DataFrame,
    y,
    n_origens: int = 24,
) -> pd.DataFrame:
    """Origem movel com re-treino para modelos supervisionados de 1 passo.

    `X` ja contem as defasagens do alvo, entao cada linha e uma previsao
    legitima de um passo a frente. Modelos multi-passo ficam para depois:
    prever varios passos exige recursao e o teste de 1 passo nao serve.
    """
    y = np.asarray(y, dtype=float).ravel()
    if len(X) != len(y):
        raise ValueError(f"X tem {len(X)} linhas e y tem {len(y)}")

    n = len(y)
    n_origens = _limitar_origens(n, n_origens)

    linhas = []
    for origem in range(n - n_origens, n):
        modelo = criar_modelo()
        modelo.fit(X.iloc[:origem], y[:origem])
        previsto = float(np.asarray(modelo.predict(X.iloc[[origem]])).ravel()[0])
        linhas.append(
            {
                "origem": origem,
                "passo": 1,
                "real": y[origem],
                "previsto": previsto,
            }
        )
    return pd.DataFrame(linhas)


def resumo(
    backtest: pd.DataFrame,
    y_treino,
    season: int = 1,
    season_sazonal: int | None = None,
) -> pd.DataFrame:
    """MAE, RMSE, vies e MASE por passo mais a linha 'geral'.

    `season=1` e o padrao de proposito: o MASE passa a ser comparavel com o
    random walk de um passo, que e o benchmark que importa aqui. O MASE com a
    diferenca sazonal fica em `MASE_sazonal`, so como informacao — com
    `season=12` o denominador vira a variacao anual e quase todo modelo
    "bate" o random walk por construcao.
    """
    if backtest.empty:
        raise ValueError("backtest vazio: nenhuma origem avaliada")

    escala = _escala_naive(y_treino, season)
    escala_sazonal = _escala_naive(y_treino, season_sazonal) if season_sazonal else None
    bt = backtest.assign(
        erro=backtest["previsto"] - backtest["real"],
        erro_abs=lambda d: d["erro"].abs(),
    )

    def _agregar(grupo: pd.DataFrame) -> pd.Series:
        mae = float(grupo["erro_abs"].mean())
        linha = {
            "MAE": mae,
            "RMSE": float(np.sqrt(np.mean(grupo["erro"] ** 2))),
            "vies": float(grupo["erro"].mean()),
            "MASE": mae / escala if escala > 0 else np.inf,
        }
        if escala_sazonal is not None:
            linha["MASE_sazonal"] = mae / escala_sazonal if escala_sazonal > 0 else np.inf
        return pd.Series(linha)

    por_passo = bt.groupby("passo").apply(_agregar, include_groups=False)
    geral = _agregar(bt).rename("geral").to_frame().T
    tabela = pd.concat([por_passo, geral])
    tabela.index.name = "passo"
    colunas = COLUNAS_RESUMO + (["MASE_sazonal"] if escala_sazonal is not None else [])
    return tabela[colunas]


def erros_absolutos(backtest: pd.DataFrame) -> np.ndarray:
    """Serie de erros absolutos out-of-fold, usada pela calibracao conformal."""
    return np.abs(
        backtest["previsto"].to_numpy(dtype=float) - backtest["real"].to_numpy(dtype=float)
    )


def diebold_mariano(erro_modelo, erro_benchmark, horizonte: int = 1) -> tuple[float, float]:
    """Teste DM com correcao Newey-West. Retorna (estatistica, p-valor)."""
    from scipy import stats

    d = np.asarray(erro_modelo, dtype=float) ** 2 - np.asarray(erro_benchmark, dtype=float) ** 2
    n = len(d)
    if n < 3:
        return 0.0, 1.0

    media_d = float(d.mean())
    variancia = float(np.var(d, ddof=1))
    for lag in range(1, min(horizonte, n - 1)):
        covariancia = float(np.cov(d[lag:], d[:-lag])[0, 1])
        variancia += 2 * (1 - lag / horizonte) * covariancia

    if variancia <= 0:
        return 0.0, 1.0
    estatistica = media_d / np.sqrt(variancia / n)
    p_valor = float(2 * (1 - stats.norm.cdf(abs(estatistica))))
    return float(estatistica), p_valor


def comparar_com_benchmark(
    backtest_modelo: pd.DataFrame,
    backtest_benchmark: pd.DataFrame,
    y_treino,
    season: int = 1,
    nome: Optional[str] = None,
) -> dict:
    """Resumo + DM contra o random walk, pronto para o JSON de metricas."""
    resultado = resumo(backtest_modelo, y_treino, season).loc["geral"].to_dict()
    try:
        estatistica, p_valor = diebold_mariano(
            backtest_modelo["previsto"].to_numpy() - backtest_modelo["real"].to_numpy(),
            backtest_benchmark["previsto"].to_numpy() - backtest_benchmark["real"].to_numpy(),
        )
        resultado["dm_estatistica"] = estatistica
        resultado["dm_p_valor"] = p_valor
    except (ValueError, KeyError):
        resultado["dm_estatistica"] = None
        resultado["dm_p_valor"] = None
    if nome:
        resultado["modelo"] = nome
    return resultado
