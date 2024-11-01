import sys

sys.path.append("..")
import pandas as pd
from bcb import sgs
import sidrapy
from bcb import Expectativas
import ipeadatapy as ip
from pytrends.request import TrendReq
import time
from datetime import date
import quandl


# Dados BCB
SELIC_CODES = {
    "selic": 4189,
    "IPCA-EX2": 27838,
    "IPCA-EX3": 27839,
    "IPCA-MS": 4466,
    "IPCA-MA": 11426,
    "IPCA-EX0": 11427,
    "IPCA-EX1": 16121,
    "IPCA-DP": 16122,
}

DATA_INICIO = "2000-01-01"


def dados_bcb(codigos_banco_central=None, data_inicio="2000-01-01"):
    dados = pd.DataFrame()
    if codigos_banco_central is None:
        codigos_banco_central = SELIC_CODES
    dados = sgs.get(codigos_banco_central, start=data_inicio)
    return dados


# DADOS IBGE


def dados_ibge_codigos(
    codigo="1737",
    territorial_level="1",
    ibge_territorial_code="all",
    variable="63",
    period="all",
):
    ipca = sidrapy.get_table(
        table_code=codigo,
        territorial_level=territorial_level,
        ibge_territorial_code=ibge_territorial_code,
        variable=variable,
        period=period,
    )
    return ipca


def dados_ibge_link(
    cabecalho=3,
    url="https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93405/d/v6561%201/l/v,p%2Bc11255,t",
):
    # carregar a tabela em um DataFrame
    dados_link = pd.read_excel(url, header=cabecalho)
    return dados_link


# Dados Expectativas/Focus


def dados_expectativas_focus(
    indicador="IPCA",
    tipo_expectativa="ExpectativaMercadoMensais",
    data_inicio="2000-01-01",
):
    # End point
    em = Expectativas()
    ep = em.get_endpoint(tipo_expectativa)

    # Dados do IPCA

    ipca_expec = (
        ep.query()
        .filter(ep.Indicador == indicador)
        .filter(ep.Data >= data_inicio)
        .filter(ep.baseCalculo == 0)
        .select(
            ep.Indicador,
            ep.Data,
            ep.Media,
            ep.Mediana,
            ep.DataReferencia,
            ep.baseCalculo,
        )
        .collect()
    )
    return ipca_expec


def dados_ipeadata(codigo="ANBIMA12_TJTLN1212", data="2020-01-01"):
    dados_ipea = ip.timeseries(codigo, yearGreaterThan=int(data[0:4]) - 1)
    return dados_ipea


def coleta_google_trends(kw_list=None, start_date=None, end_date=None):
    if start_date is None:
        start_date = "2004-01-01"
    if end_date is None:
        end_date = str(date.today())
    if kw_list is None:
        kw_list = ["seguro desemprego"]

    # Dividindo a pesquisa em vários períodos de 5 anos
    periods = pd.date_range(start=start_date, end=end_date, freq="5YE")
    if periods[-1] < pd.to_datetime(end_date):
        periods = periods.append(pd.Index([pd.to_datetime(end_date)]))

    # Criando uma instância do TrendReq
    pytrends = TrendReq()

    # Fazendo a pesquisa para cada período de 5 anos
    data = pd.DataFrame()
    for i in range(len(periods) - 1):
        timeframe = (
            f"{periods[i].strftime('%Y-%m-%d')} {periods[i+1].strftime('%Y-%m-%d')}"
        )
        pytrends.build_payload(kw_list, timeframe=timeframe)
        temp_data = pytrends.interest_over_time()
        data = pd.concat([data, temp_data])

        # Aguardando alguns segundos para evitar sobrecarga no servidor do Google Trends
        time.sleep(10)
    print("O google trends tem como data de início o ano 2004.")
    return data


def coleta_quandl(codes=None):
    data = quandl.get(codes)
    return data
