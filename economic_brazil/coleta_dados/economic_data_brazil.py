from .tratando_economic_brazil import (
    tratando_dados_bcb,
    tratando_dados_expectativas,
    tratando_dados_ibge_link,
    tratando_dados_ibge_codigos,
    tratando_metas_inflacao,
)
import pandas as pd
from datetime import datetime
import warnings
from functools import lru_cache

warnings.filterwarnings("ignore")

DATA_INICIO = "2000-01-01"

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


def fetch_data_for_code(link, column):
    return tratando_dados_ibge_link(coluna=column, link=link)


@lru_cache(maxsize=50)
def data_economic(
    codigos_banco_central=None,
    data_inicio=DATA_INICIO,
    salvar=False,
    diretorio=None,
    formato="csv",
    **kwargs,
):
    data_index = pd.date_range(
        start=data_inicio, end=datetime.today().strftime("%Y-%m-%d"), freq="MS"
    )
    dados = pd.DataFrame(index=data_index)
    try:
        if kwargs.get("banco_central", True):
            if codigos_banco_central is None:
                codigos_banco_central = SELIC_CODES
            dados = tratando_dados_bcb(
                codigo_bcb_tratado=codigos_banco_central,
                data_inicio_tratada=data_inicio,
            )

        if kwargs.get("expectativas_inflacao", True):
            dados["expectativas_inflacao"] = tratando_dados_expectativas()

        if kwargs.get("meta_inflacao", True):
            dados = dados.join(
                tratando_metas_inflacao(),
            )

        if kwargs.get("ipca", True):
            dados["ipca"] = tratando_dados_ibge_codigos()["Valor"]

        indicadores = {
            "pib": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/90707/d/v6561%201/l/v,p%2Bc11255,t",
            "despesas_publica": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93405/d/v6561%201/l/v,p%2Bc11255,t",
            "capital_fixo": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93406/d/v6561%201/l/v,p%2Bc11255,t",
            "producao_industrial_manufatureira": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela8158.xlsx&terr=N&rank=-&query=t/8158/n1/all/v/11599/p/all/c543/129278/d/v11599%205/l/v,p%2Bc543,t",
        }

        for key, link in indicadores.items():
            if kwargs.get(key, True):
                dados[key] = fetch_data_for_code(link, key)
            else:
                print(f"Dados para '{key}' não solicitados.")

        dado_sem_nan = dados.ffill()
        dado_sem_nan = dado_sem_nan.bfill()

    except ValueError as _:
        dado_sem_nan = pd.DataFrame(
            "/workspaces/Predicoes_macroeconomicas/dados/economic_data_brazil.csv"
        )
        ultima_data = dados.index[-1]
        print(
            f"Problema na importação dos dados.Arquivo selecionado da memoria com a ultima data sendo {ultima_data}."
        )

    if salvar:
        if diretorio is None:
            raise ValueError("Diretório não especificado para salvar o arquivo")
        if formato == "csv":
            dado_sem_nan.to_csv(diretorio)
        elif formato == "excel":
            dado_sem_nan.to_excel(f"{diretorio}.xlsx")
        elif formato == "json":
            dado_sem_nan.to_json(f"{diretorio}.json")
        else:
            raise ValueError("Formato de arquivo não suportado")

    return dado_sem_nan