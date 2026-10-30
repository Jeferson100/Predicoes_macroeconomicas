"""Backtest, MASE e conformal: as metricas que decidem publicacao."""

import numpy as np
import pandas as pd
import pytest

from economic_brazil.forecasting import backtest, baselines, conformal
from economic_brazil.forecasting.features import montar_dataset, montar_linha_futura


@pytest.fixture
def serie() -> pd.Series:
    indice = pd.date_range("2000-01-01", periods=240, freq="MS")
    rng = np.random.default_rng(0)
    return pd.Series(10 + np.cumsum(rng.normal(0, 0.3, len(indice))), index=indice)


@pytest.fixture
def ruido_branco() -> pd.Series:
    """Serie sem tendencia: o random walk e a escala ficam na mesma janela."""
    indice = pd.date_range("2000-01-01", periods=240, freq="MS")
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0, 1, len(indice)), index=indice)


def test_random_walk_tem_mase_proximo_de_um(ruido_branco):
    """O random walk e a escala do MASE, entao o MASE dele e ~1."""
    bt = backtest.backtest_baseline(baselines.random_walk, ruido_branco, n_origens=24)
    tabela = backtest.resumo(bt, ruido_branco[:-24], season=1)

    assert tabela.loc["geral", "MASE"] == pytest.approx(1.0, abs=0.25)


def test_mase_e_mae_dividido_pela_escala(serie):
    """A definicao precisa ficar explicita para o gate nao mudar de sentido."""
    bt = backtest.backtest_baseline(baselines.random_walk, serie, n_origens=24)
    tabela = backtest.resumo(bt, serie[:-24], season=1)
    escala = float(np.mean(np.abs(np.diff(serie[:-24].to_numpy()))))

    assert tabela.loc["geral", "MASE"] == pytest.approx(tabela.loc["geral", "MAE"] / escala)


def test_mase_sazonal_fica_separado_do_mase_principal(serie):
    """Com season=12 o denominador e a variacao anual: nao serve de gate."""
    bt = backtest.backtest_baseline(baselines.random_walk, serie, n_origens=24)
    tabela = backtest.resumo(bt, serie[:-24], season=1, season_sazonal=12)

    assert "MASE_sazonal" in tabela.columns
    assert tabela.loc["geral", "MASE_sazonal"] < tabela.loc["geral", "MASE"]


def test_resumo_rejeita_backtest_vazio():
    with pytest.raises(ValueError, match="vazio"):
        backtest.resumo(pd.DataFrame(columns=["previsto", "real", "passo"]), [1.0, 2.0])


def test_cobertura_conformal_usa_a_janela_de_avaliacao(serie):
    """Antes a cobertura retornava None quando n_calib >= metade dos erros."""
    bt = backtest.backtest_baseline(baselines.random_walk, serie, n_origens=24)
    cobertura = conformal.cobertura_empirica(bt, alpha=0.2, n_calib=24)

    assert cobertura is not None
    assert 0.0 <= cobertura <= 1.0


def test_intervalos_respeitam_o_quantil():
    erros = np.arange(1, 101, dtype=float)
    intervalo = conformal.intervalos(0.0, erros, alphas=(0.2,))["80%"]

    assert intervalo[0] == pytest.approx(-81.0)
    assert intervalo[1] == pytest.approx(81.0)


def test_montar_linha_futura_nao_ve_o_futuro(serie):
    """A linha futura so usa valores passados e cai no periodo seguinte."""
    X, _ = montar_dataset(serie, lags=(1, 2), sazonalidade=False)
    linha = montar_linha_futura(serie, lags=(1, 2), sazonalidade=False)

    assert linha.index[0] == serie.index[-1] + pd.DateOffset(months=1)
    assert list(linha.columns) == list(X.columns)
    assert linha.iloc[0]["alvo_lag_1"] == serie.iloc[-1]
    assert linha.iloc[0]["alvo_lag_2"] == serie.iloc[-2]
