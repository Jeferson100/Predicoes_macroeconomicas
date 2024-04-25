from .coleta_economic_brazil import (
    dados_bcb,
    dados_ibge_codigos,
    dados_expectativas_focus,
    dados_ibge_link,
    metas_inflacao,
)
import pandas as pd
import numpy as np
from datetime import datetime

# Tratando dados IBGE/SIDRAPY


def trimestral_para_mensal(df):
    """
    A função recebe um DataFrame df com valores trimestrais do PIB. Primeiro, ela aplica a interpolação para obter os valores mensais, usando o método resample com uma frequência de 'M' e o método interpolate para preencher os valores faltantes.
    Em seguida, ela percorre os valores de cada trimestre e distribui a variação trimestral em cada um dos três meses dentro do trimestre, adicionando um terço do valor trimestral aos dois meses intermediários. Finalmente, ela retorna o DataFrame
    interpolado e transformado em mensal.
    """

    # Primeiro, definimos uma nova frequência mensal e aplicamos a interpolação
    df_mensal = df.resample("MS").interpolate()

    # Em seguida, distribuímos a variação trimestral em cada um dos três meses dentro do trimestre
    for i in range(1, len(df)):
        if i % 3 != 0:
            val = df.iloc[i].values[0]
            month_val = val / 3
            df_mensal.iloc[i * 2 - 1] += month_val
            df_mensal.iloc[i * 2] += month_val

    return df_mensal


def converter_mes_para_data(mes):
    mes_texto = str(mes)
    ano = int(mes_texto[:4])
    mes = int(mes_texto[4:])
    data = datetime(year=ano, month=mes, day=1)
    return data


def trimestre_string_int(dados):
    lista_trimestre = []
    for i in range(len(dados.index)):
        lista_trimestre.append(dados.index[i][-4:] + "-" + "0" + dados.index[i][0])
    return lista_trimestre


def transforma_para_mes_incial_trimestre(dados):
    lista_mes = []
    for i in range(len(dados.index)):
        trimestre = dados.index.month[i]
        ano = str(dados.index.year[i])
        lista_mes.append(
            str(
                np.where(
                    trimestre == 1,
                    ano + "-" + "0" + str(trimestre),
                    np.where(
                        trimestre == 2,
                        ano + "-" + "0" + str(trimestre + 2),
                        np.where(
                            trimestre == 3,
                            ano + "-" + "0" + str(trimestre + 4),
                            np.where(trimestre == 4, ano + "-" + str(trimestre + 6), 0),
                        ),
                    ),
                )
            )
        )
    return lista_mes


def transforme_data(data):
    months = {
        "janeiro": "january",
        "fevereiro": "february",
        "março": "march",
        "abril": "april",
        "maio": "may",
        "junho": "june",
        "julho": "july",
        "agosto": "august",
        "setembro": "september",
        "outubro": "october",
        "novembro": "november",
        "dezembro": "december",
    }

    lista_data = []
    for date in data.index:
        date_components = date.split(" ")
        formatted_month = months[date_components[0].lower()].capitalize()
        formatted_date = f"{formatted_month} {date_components[1]}"
        date_object = datetime.strptime(formatted_date, "%B %Y")
        lista_data.append(date_object.strftime("%Y-%m-%d"))
    data.index = lista_data
    return data


###Tratando dados IBGE


def tratando_dados_ibge_codigos(salvar=False, formato="csv", diretorio=None):
    ibge_codigos = dados_ibge_codigos()
    ibge_codigos.columns = ibge_codigos.iloc[0, :]
    ibge_codigos = ibge_codigos.iloc[1:, :]
    ibge_codigos["data"] = ibge_codigos["Mês (Código)"].apply(converter_mes_para_data)
    ibge_codigos.index = ibge_codigos["data"]
    ibge_codigos["Valor"] = ibge_codigos["Valor"][1:].astype(float)
    if salvar:
        if diretorio is None:
            raise ValueError("Diretório não especificado para salvar o arquivo")
        if formato == "csv":
            ibge_codigos.to_csv(diretorio)
        elif formato == "json":
            ibge_codigos.to_json(diretorio)
    return ibge_codigos


def tratando_dados_ibge_link(
    coluna="pib",
    link="",
    salvar=False,
    formato="csv",
    diretorio=None,
):
    dado_ibge = dados_ibge_link(url=link)
    ibge_link = dado_ibge.T
    ibge_link = ibge_link[[1]]
    ibge_link = ibge_link[1:]
    ibge_link.columns = [coluna]
    ibge_link[coluna] = pd.to_numeric(ibge_link[coluna], errors="coerce")
    if coluna == "producao_industrial_manufatureira":
        ibge_link = transforme_data(ibge_link)
        ibge_link.index = pd.to_datetime(ibge_link.index)
    else:
        ibge_link.index = pd.to_datetime(trimestre_string_int(ibge_link))

        ibge_link.index = pd.to_datetime(
            transforma_para_mes_incial_trimestre(ibge_link)
        )
        # ibge_link = ibge_link.resample("MS").fillna(method="ffill")
        ibge_link = ibge_link.resample("MS").ffill()

    if salvar:
        if diretorio is None:
            raise ValueError("Diretório não especificado para salvar o arquivo")
        if formato == "csv":
            ibge_link.to_csv(diretorio)
        elif formato == "json":
            ibge_link.to_json(diretorio)

    return ibge_link


###Tratando dados BCB
selic = {
    "selic": 4189,
    "IPCA-EX2": 27838,
    "IPCA-EX3": 27839,
    "IPCA-MS": 4466,
    "IPCA-MA": 11426,
    "IPCA-EX0": 11427,
    "IPCA-EX1": 16121,
    "IPCA-DP": 16122,
}


def tratando_dados_bcb(
    codigo_bcb_tratado=None,
    data_inicio_tratada="2000-01-01",
    salvar=False,
    diretorio=None,
    formato="csv",
    **kwargs,
):
    if codigo_bcb_tratado is None:
        codigo_bcb_tratado = selic
    if not isinstance(codigo_bcb_tratado, dict):
        print("Código BCB deve ser um dicionário. Usando valor padrão.")
        codigo_bcb_tratado = selic
    inflacao_bcb = dados_bcb(codigo_bcb_tratado, data_inicio_tratada, **kwargs)
    if salvar:
        if diretorio is None:
            raise ValueError("Diretório não especificado para salvar o arquivo")
        if formato == "csv":
            inflacao_bcb.to_csv(diretorio)
        elif formato == "json":
            inflacao_bcb.to_json(diretorio)
    return inflacao_bcb


###Tratando dados Expectativas


def tratando_dados_expectativas(salvar=False, formato="csv", diretorio=None):
    ipca_expec = dados_expectativas_focus()
    dados_ipca = ipca_expec.copy()
    dados_ipca = dados_ipca[::-1]
    dados_ipca["monthyear"] = pd.to_datetime(dados_ipca["Data"]).apply(
        lambda x: x.strftime("%Y-%m")
    )
    dados_ipca = dados_ipca.groupby("monthyear")["Mediana"].mean()
    # criar índice com o formato "YYYY-MM"
    dados_ipca.index = pd.to_datetime(dados_ipca.index, format="%Y-%m")

    # adicionar o dia como "01"
    dados_ipca.index = dados_ipca.index.to_period("M").to_timestamp()

    if salvar:
        if diretorio is None:
            raise ValueError("Diretório não especificado para salvar o arquivo")
        if formato == "csv":
            dados_ipca.to_csv(diretorio)
        elif formato == "json":
            dados_ipca.to_json(diretorio)

    return dados_ipca


## Tratando metas de inflação


def tratando_metas_inflacao(salvar=False, formato="csv", diretorio=None):
    historico_inflacao = metas_inflacao()
    duplicado_ano_2000 = int(
        historico_inflacao[historico_inflacao["anos"].duplicated()]["anos"].iloc[0]
    )
    if duplicado_ano_2000 == 2000:
        primeira_ocorrencia = historico_inflacao["anos"].duplicated(keep="first")
        historico_inflacao.loc[primeira_ocorrencia, "anos"] = "2001"
    historico_inflacao.index = pd.to_datetime(
        historico_inflacao["anos"].str.strip(), format="%Y"
    )
    historico_inflacao.drop("anos", axis=1, inplace=True)
    historico_inflacao = historico_inflacao.resample("MS").ffill()

    if salvar:
        if diretorio is None:
            raise ValueError("Diretório não especificado para salvar o arquivo")
        if formato == "csv":
            historico_inflacao.to_csv(diretorio)
        elif formato == "json":
            historico_inflacao.to_json(diretorio)

    return historico_inflacao
