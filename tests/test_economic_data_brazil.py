import pytest
from unittest.mock import patch, mock_open
import pandas as pd
from economic_brazil.coleta_dados.economic_data_brazil import EconomicBrazil

# pylint: disable=W0621
@pytest.fixture
def econ_brazil():
    # dados_economicos = EconomicBrazil()
    return EconomicBrazil()


# pylint: disable=W0621


def test_dados_banco_central(econ_brazil):
    # dados_economicos = EconomicBrazil()
    dados = econ_brazil.dados_banco_central()
    assert isinstance(dados.index, pd.DatetimeIndex)
    assert isinstance(dados, pd.DataFrame)


def test_dados_expectativas_inflacao(econ_brazil):
    dados = econ_brazil.dados_expectativas_inflacao()
    assert isinstance(dados.index, pd.DatetimeIndex)
    assert isinstance(dados, pd.DataFrame)


def test_dados_metas_inflacao(econ_brazil):
    dados = econ_brazil.dados_metas_inflacao()
    assert isinstance(dados.index, pd.DatetimeIndex)
    assert isinstance(dados, pd.DataFrame)


def test_dados_ibge(econ_brazil):
    dados = econ_brazil.dados_ibge()
    assert isinstance(dados.index, pd.DatetimeIndex)
    assert isinstance(dados, pd.DataFrame)


def test_dados_ibge_link(econ_brazil):
    dados = econ_brazil.dados_ibge_link()
    assert isinstance(dados.index, pd.DatetimeIndex)
    assert isinstance(dados, pd.DataFrame)


def test_dados_ipeadata(econ_brazil):
    dados = econ_brazil.dados_ipeadata()
    assert isinstance(dados.index, pd.DatetimeIndex)
    assert isinstance(dados, pd.DataFrame)


def test_dados_google_trends(econ_brazil):
    dados = econ_brazil.dados_google_trends()
    assert isinstance(dados.index, pd.DatetimeIndex)
    assert isinstance(dados, pd.DataFrame)


def test_dados_dados_brazil(econ_brazil):
    dados = econ_brazil.dados_brazil()
    assert not dados.isna().any().any()
    assert isinstance(dados.index, pd.DatetimeIndex)
    assert isinstance(dados, pd.DataFrame)


def test_salvar_dados(econ_brazil):
    dados = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    with patch("builtins.open", mock_open()) as mocked_file:
        econ_brazil.salvar_dados(dados, diretorio="test_file", formato="pickle")
        mocked_file.assert_called_with("test_file.pkl", "wb")
