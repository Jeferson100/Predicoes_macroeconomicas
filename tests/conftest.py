"""Fixtures sinteticas: nenhum teste offline toca a rede."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dados_sinteticos() -> pd.DataFrame:
    """DataFrame com o mesmo contrato do real, sem rede."""
    indice = pd.date_range("2010-01-01", periods=180, freq="MS")
    rng = np.random.default_rng(42)

    dados = pd.DataFrame(index=indice)
    dados["selic"] = 10 + np.cumsum(rng.normal(0, 0.2, len(indice)))
    for i in range(8):
        dados[f"feature_{i}"] = rng.normal(0, 1, len(indice)).cumsum()
    return dados


@pytest.fixture
def dados_trimestrais(dados_sinteticos: pd.DataFrame) -> pd.DataFrame:
    return dados_sinteticos.resample("QS").mean()
