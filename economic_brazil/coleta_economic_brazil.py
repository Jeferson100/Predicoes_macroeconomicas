import pandas as pd
from bcb import sgs
import sidrapy
from bcb import Expectativas
from bs4 import BeautifulSoup
import requests
import math


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


# Metas de inflacao


def metas_inflacao():
    # Load the web page and extract the table contents
    url = (
        "https://www.bcb.gov.br/api/paginasite/sitebcb/controleinflacao/historicometas"
    )
    page = requests.get(url, timeout=10)
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find_all("table")[0]

    # Extract the data and inflation target columns and remove unwanted rows
    td = table.find_all("td")
    # Define regular expressions to extract the year and inflation target values
    # year_regex = r"<td>(\d{4})<br/></td>"
    # year_tags = re.findall(year_regex, str(td))

    ##Pergar meta de inflacao
    meta_inflacao = []
    ano_inflacao = []
    inflacao_efetiva = []
    index = 0
    for _ in range(3):
        meta_inflacao.append(td[8 + index].get_text())
        ano_inflacao.append(td[7 + index].get_text())
        inflacao_efetiva.append(td[11 + index].get_text())
        index = index + 5
        index = 0
    for _ in range(math.ceil((len(td) - 20) / 7)):
        meta_inflacao.append(td[20 + index].get_text())
        ano_inflacao.append(td[17 + index].get_text())
        inflacao_efetiva.append(td[23 + index].get_text())
        index = index + 7
        historico_inflacao = pd.DataFrame()
        historico_inflacao["meta_inflacao"] = meta_inflacao
        historico_inflacao["anos"] = ano_inflacao
        historico_inflacao["inflacao_efetiva"] = inflacao_efetiva
    # Apagando a linha 2
    historico_inflacao.drop(2, axis=0, inplace=True)
    historico_inflacao.loc[4, "inflacao_efetiva"] = historico_inflacao.loc[
        4, "inflacao_efetiva"
    ][:3]
    historico_inflacao.loc[5, "meta_inflacao"] = historico_inflacao.loc[
        5, "meta_inflacao"
    ][:3]
    historico_inflacao.loc[0, "meta_inflacao"] = historico_inflacao.loc[
        0, "meta_inflacao"
    ][-1]
    historico_inflacao.loc[21, "meta_inflacao"] = historico_inflacao.loc[
        21, "meta_inflacao"
    ][-1]

    # Funcao para tirar valores \u200b
    funcao = [lambda x: x.replace("\u200b", "")]
    funcao_2 = [lambda x: x.replace(",", ".")]
    funcao_3 = [lambda x: x.replace("*", "")]
    # Arruma a coluna meta inflacao
    historico_inflacao["meta_inflacao"] = (
        historico_inflacao["meta_inflacao"].apply(funcao).apply(funcao_2).astype(float)
    )
    # Arruma coluna anos
    historico_inflacao["anos"] = historico_inflacao["anos"].apply(funcao_3)
    # Coverte em flutuante a coluna inflacao_efetiva
    historico_inflacao["inflacao_efetiva"] = (
        historico_inflacao["inflacao_efetiva"].apply(funcao).apply(funcao_2)
    )
    historico_inflacao["inflacao_efetiva"] = pd.to_numeric(
        historico_inflacao["inflacao_efetiva"], errors="coerce"
    ).tolist()
    historico_inflacao["diferenca_meta_efetiva"] = (
        historico_inflacao["meta_inflacao"] - historico_inflacao["inflacao_efetiva"]
    )

    return historico_inflacao
