from economic_brazil.coleta_dados.tratando_economic_brazil import (
    tratando_dados_expectativas,
    tratando_dados_ibge_codigos,
    tratando_dados_ibge_link,
    tratando_metas_inflacao,
    tratatando_dados_ipeadata,
)
import pandas as pd


# write tests for tratando_dados_bcb,
# pylint: disable=W0105

"""def test_tratando_dados_bcb_datetime():
    dados = tratando_dados_bcb({"selic": 4189}, "2000-01-01")
    assert isinstance(dados.index, pd.DatetimeIndex)
"""
# pylint: disable=W0105

# pylint: disable=W0105
"""def test_tratando_dados_bcb_columns():
    dados = tratando_dados_bcb({"selic": 4189}, "2000-01-01")
    assert "selic" in dados.columns"""
# pylint: disable=W0105


# write tests for tratando_dados_expectativas
def test_tratando_dados_expectativas():
    dados = tratando_dados_expectativas()
    assert isinstance(dados.index, pd.DatetimeIndex)


# write tests for tratando_dados_ibge_codigos
def test_tratando_dados_ibge_codigos_time():
    dados = tratando_dados_ibge_codigos()
    assert isinstance(dados.index, pd.DatetimeIndex)


def test_tratando_dados_ibge_codigos_columns():
    dados = tratando_dados_ibge_codigos()
    assert "Valor" in dados.columns


# write tests for tratando_dados_ibge_link
def test_tratando_dados_ibge_link():
    dados = tratando_dados_ibge_link(
        coluna="pib",
        link="https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/90707/d/v6561%201/l/v,p%2Bc11255,t",
    )
    assert isinstance(dados.index, pd.DatetimeIndex)


def test_tratando_dados_ibge_link_columns():
    dados = tratando_dados_ibge_link(
        coluna="pib",
        link="https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/90707/d/v6561%201/l/v,p%2Bc11255,t",
    )
    assert "pib" in dados.columns


# write tests for tratando_metas_inflacao
def test_tratando_metas_inflacao_time():
    dados = tratando_metas_inflacao()
    assert isinstance(dados.index, pd.DatetimeIndex)


def test_tratando_metas_inflacao_columns():
    dados = tratando_metas_inflacao()
    assert ["meta_inflacao", "inflacao_efetiva", "diferenca_meta_efetiva"] == list(
        dados.columns
    )


def test_tratatando_dados_ipeadata():
    codigo_ipea = {"taja_juros_ltn": "ANBIMA12_TJTLN1212"}
    dados = tratatando_dados_ipeadata(codigo_ipeadata=codigo_ipea, data="2000-01-01")
    assert isinstance(dados.index, pd.DatetimeIndex)
    assert not dados.empty
