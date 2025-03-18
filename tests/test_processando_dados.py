import sys

sys.path.append("..")
import pandas as pd
import numpy as np
from economic_brazil.processando_dados.data_processing import (
    criando_dummy_covid,
    backcasting_nan,
    criando_defasagens,
    corrigindo_nan_arima,
    criando_mes_ano_dia,
    escalando_dados,
)


def test_criando_dummy_covid() -> None:
    dates = pd.date_range(start="2020-01-01", end="2020-12-31")
    data = pd.DataFrame(index=dates)
    data = criando_dummy_covid(data, "2020-03-01", "2020-06-30")
    assert "dummy_covid" in data.columns
    assert data["dummy_covid"].sum() == len(
        pd.date_range(start="2020-03-01", end="2020-06-30")
    )
    assert data["dummy_covid"].iloc[0] == 0
    assert data["dummy_covid"].loc["2020-03-01"] == 1
    assert data["dummy_covid"].loc["2020-07-01"] == 0


def test_backcasting_nan() -> None:
    data = pd.DataFrame(
        {
            "A": [1, np.nan, 3, np.nan, 5],
            "B": [1, 2, np.nan, 4, 5],
            "C": [1, np.nan, np.nan, 4, 5],
        }
    )
    result = backcasting_nan(data)
    assert not result.isnull().values.any()


def test_criando_defasagens() -> None:
    data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})
    result = criando_defasagens(data, numero_defasagens=2)
    assert "A_lags_1" in result.columns
    assert "B_lags_2" in result.columns
    assert result["A_lags_1"].iloc[1] == 1
    assert result["B_lags_2"].iloc[2] == 5


def test_corrigindo_nan_arima() -> None:
    data = pd.DataFrame(
        {
            "A": [1, 2, np.nan, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan],
            "B": [5, np.nan, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan],
        }
    )
    result = corrigindo_nan_arima(data, "A")
    assert not result.isnull().values.any()


def test_criando_mes_ano_dia() -> None:
    dates = pd.date_range(start="2020-01-01", periods=5)
    data = pd.DataFrame(index=dates)
    result = criando_mes_ano_dia(data, mes=True, ano=True, dia=True, trimestre=True)
    assert "mes" in result.columns
    assert "ano" in result.columns
    assert "dia" in result.columns
    assert "trimestre" in result.columns
    assert result["mes"].iloc[0] == 1
    assert result["ano"].iloc[0] == 2020
    assert result["dia"].iloc[0] == 1
    assert result["trimestre"].iloc[0] == 1


def test_escalando_dados() -> None:
    data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})
    scaled_data, _ = escalando_dados(data.values, tipo="minmax")
    assert scaled_data.min() == 0
    assert scaled_data.max() == 1
    scaled_data, _ = escalando_dados(data.values, tipo="standard")
    assert np.isclose(scaled_data.mean(), 0)
    assert np.isclose(scaled_data.std(), 1)
