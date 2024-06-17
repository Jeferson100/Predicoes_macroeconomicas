"""

import sys
sys.path.append("..")
from economic_brazil.coleta_dados.tratando_economic_brazil import (
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
import requests
from urllib.error import URLError
import pickle

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
    "cambio": 3698,
    "pib_mensal": 4380,
    "igp_m": 189,
    "igp_di": 190,
    "m1": 27788,
    "m2": 27810,
    "m3": 27813,
    "m4": 27815,
    'estoque_caged': 28763,
    'saldo_bc': 22707,
    'vendas_auto':7384,
    'divida_liquida_spc':4513,  
}

variaveis_ibge = {
    'ipca': {'codigo': 1737, 'territorial_level': '1', 'ibge_territorial_code': 'all', 'variable': '63'},
    'custo_m2': {'codigo': 2296, 'territorial_level': '1', 'ibge_territorial_code': 'all', 'variable': '1198'},
    'pesquisa_industrial_mensal': {'codigo': 8159, 'territorial_level': '1', 'ibge_territorial_code': 'all', 'variable': '11599'},
    'pmc_volume': {'codigo': 8186, 'territorial_level': '1', 'ibge_territorial_code': 'all', 'variable': '11709'},
}

indicadores_ibge_link = {
            "pib": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/90707/d/v6561%201/l/v,p%2Bc11255,t",
            "despesas_publica": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93405/d/v6561%201/l/v,p%2Bc11255,t",
            "capital_fixo": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93406/d/v6561%201/l/v,p%2Bc11255,t",
            "producao_industrial_manufatureira": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela8158.xlsx&terr=N&rank=-&query=t/8158/n1/all/v/11599/p/all/c543/129278/d/v11599%205/l/v,p%2Bc543,t",
        }


def fetch_data_for_code(link, column):
    return tratando_dados_ibge_link(coluna=column, link=link)


#@lru_cache(maxsize=100)
def data_economic(
    codigos_banco_central=None,
    codigos_ibge=None,
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
            try:
                dados = tratando_dados_bcb(
                    codigo_bcb_tratado=codigos_banco_central,
                    data_inicio_tratada=data_inicio,
                )
            except ValueError as e:
                print("Erro ao buscar dados", e)
                dados = pd.read_csv(
                    "/workspaces/Predicoes_macroeconomicas/dados/dados_bcb.csv"
                )
                dados.index = pd.to_datetime(dados["Date"])
                dados = dados.drop("Date", axis=1)
                ultima_data = dados.index[-1]
                print(
                    f"Problema na importação dos dados do Banco Central.Arquivo selecionado da memoria com a ultima data sendo {ultima_data}."
                )

        if kwargs.get("expectativas_inflacao", True):
            dados["expectativas_inflacao"] = tratando_dados_expectativas()

        if kwargs.get("meta_inflacao", True):
            dados = dados.join(
                tratando_metas_inflacao(),
            )

        if kwargs.get("ibge", True):
            try:
                if codigos_ibge is None:
                    codigos_ibge = variaveis_ibge       
                for key, valor in codigos_ibge.items():
                    try:
                        dados[key] = tratando_dados_ibge_codigos(codigos=valor)['Valor']
                    except ValueError:
                        print(f'Erro na coleta de dados da varivel {key}. Verifique se os codigos {valor} estão ativos.')
            except requests.exceptions.SSLError as e:
                error_message = str(e).split()[
                    :4
                ]  # Ajuste o número de palavras conforme necessário
                print(f"Erro na API IBGE: {' '.join(error_message)}")
                dados_ipca = pd.read_csv("../dados/economic_data_brazil.csv")
                dados_ipca.index = pd.to_datetime(dados_ipca.Date)
                dados["ipca"] = dados_ipca["ipca"]
                ultima_data = dados.index[-1]
                print(
                    f"Problema na importação dos dados do IBGE .Arquivo selecionado da memoria com a ultima data sendo {ultima_data}."
                )

        for key, link in indicadores_ibge_link.items():
            if kwargs.get(key, True):
                try:
                    dados[key] = fetch_data_for_code(link, key)
                except URLError as e:
                    error_message = str(e).split()[
                        :2
                    ]  # Ajuste o número de palavras conforme necessário
                    print(f"Erro na API IBGE Link: {' '.join(error_message)}")
                    dados_ibge = pd.read_csv("../dados/economic_data_brazil.csv")
                    dados_ibge.index = pd.to_datetime(dados_ipca.Date)
                    dados[key] = dados_ibge[key]
                    ultima_data = dados.index[-1]
                    print(
                        f"Problema na importação dos dados do IBGE {key}.Arquivo selecionado da memoria com a ultima data sendo {ultima_data}."
                    )
            else:
                print(f"Dados para '{key}' não solicitados.")

        dado_sem_nan = dados.ffill()
        dado_sem_nan = dado_sem_nan.bfill()

    except ValueError as e:
        print("Erro ao buscar dados", e)
        dado_sem_nan = pd.DataFrame(
            "/workspaces/Predicoes_macroeconomicas/dados/economic_data_brazil.csv"
        )
        dados.index = pd.to_datetime(dados["Date"])
        dados = dados.drop("Date", axis=1)
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
        elif formato == 'pickle':
            with open(f'{diretorio}', 'wb') as f:
                pickle.dump(dado_sem_nan, f)
        else:
            raise ValueError("Formato de arquivo não suportado")

    return dado_sem_nan

"""

import sys
import pandas as pd
from datetime import datetime
import warnings
import pickle

sys.path.append("..")
from economic_brazil.coleta_dados.tratando_economic_brazil import (
    tratando_dados_bcb,
    tratando_dados_expectativas,
    tratando_dados_ibge_link,
    tratando_dados_ibge_codigos,
    tratando_metas_inflacao,
    tratatando_dados_ipeadata,
    tratando_dados_google_trends,
)

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
    "cambio": 3698,
    "pib_mensal": 4380,
    "igp_m": 189,
    "igp_di": 190,
    "m1": 27788,
    "m2": 27810,
    "m3": 27813,
    "m4": 27815,
    "estoque_caged": 28763,
    "saldo_bc": 22707,
    "vendas_auto": 7384,
    "divida_liquida_spc": 4513,
}

variaveis_ibge = {
    "ipca": {
        "codigo": 1737,
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "63",
    },
    "custo_m2": {
        "codigo": 2296,
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "1198",
    },
    "pesquisa_industrial_mensal": {
        "codigo": 8159,
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "11599",
    },
    "pmc_volume": {
        "codigo": 8186,
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "11709",
    },
}

codigos_ipeadata_padrao = {
    "taja_juros_ltn": "ANBIMA12_TJTLN1212",
    "imposto_renda": "SRF12_IR12",
    "ibovespa": "ANBIMA12_IBVSP12",
    "consumo_energia": "ELETRO12_CEET12",
    "brent_fob": "EIA366_PBRENT366",
}

indicadores_ibge_link = {
    "pib": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/90707/d/v6561%201/l/v,p%2Bc11255,t",
    "despesas_publica": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93405/d/v6561%201/l/v,p%2Bc11255,t",
    "capital_fixo": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93406/d/v6561%201/l/v,p%2Bc11255,t",
    "producao_industrial_manufatureira": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela8158.xlsx&terr=N&rank=-&query=t/8158/n1/all/v/11599/p/all/c543/129278/d/v11599%205/l/v,p%2Bc543,t",
}

lista_google_trends = [
    "seguro desemprego",
]


class EconomicBrazil:
    def __init__(
        self,
        codigos_banco_central=None,
        codigos_ibge=None,
        codigos_ibge_link=None,
        codigos_ipeadata=None,
        lista_termos_google_trends=None,
        data_inicio=None,
    ):
        self.codigos_banco_central = codigos_banco_central or SELIC_CODES
        self.codigos_ibge = codigos_ibge or variaveis_ibge
        self.codigos_ibge_link = codigos_ibge_link or indicadores_ibge_link
        self.codigos_ipeadata = codigos_ipeadata or codigos_ipeadata_padrao
        self.data_inicio = data_inicio or DATA_INICIO
        self.lista_termos_google_trends = (
            lista_termos_google_trends or lista_google_trends
        )

    def fetch_data_for_code(self, link, column):
        return tratando_dados_ibge_link(coluna=column, link=link)

    def data_index(self):
        data_index = pd.date_range(
            start=self.data_inicio, end=datetime.today().strftime("%Y-%m-%d"), freq="MS"
        )
        return pd.DataFrame(index=data_index)

    def dados_banco_central(self, salvar=None, diretorio=None, formato="csv"):
        dados = pd.DataFrame()
        for nome, codigo in self.codigos_banco_central.items():
            try:
                dados[nome] = tratando_dados_bcb(
                    codigo_bcb_tratado={nome: codigo},
                    data_inicio_tratada=self.data_inicio,
                )[nome]
                if dados[nome].dtype == "object":
                    dados[nome] = pd.to_numeric(dados[nome], errors="coerce")
            except ValueError:
                print(
                    f"Erro na coleta de dados da variável {nome}. Verifique se o código {codigo} está ativo https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do?method=prepararTelaLocalizarSeries"
                )
        if salvar:
            self.salvar_dados(dados, diretorio, formato)
        else:
            return dados

    def dados_expectativas_inflacao(self, salvar=None, diretorio=None, formato="csv"):
        dic_expectativas_inflacao = self.data_index()
        dic_expectativas_inflacao = dic_expectativas_inflacao.join(
            tratando_dados_expectativas()
        )
        if "Mediana" in dic_expectativas_inflacao.columns:
            dic_expectativas_inflacao.rename(
                columns={"Mediana": "ipca_expectativa_focus"}, inplace=True
            )
        if salvar:
            self.salvar_dados(dic_expectativas_inflacao, diretorio, formato)
        else:
            return dic_expectativas_inflacao

    def dados_metas_inflacao(self, salvar=None, diretorio=None, formato="csv"):
        dic_metas_inflacao = self.data_index()
        dic_metas_inflacao = dic_metas_inflacao.join(tratando_metas_inflacao())
        if salvar:
            self.salvar_dados(dic_metas_inflacao, diretorio, formato)
        else:
            return dic_metas_inflacao

    def dados_ibge(self, salvar=False, diretorio=None, formato="pickle"):
        dic_ibge = self.data_index()
        for key, valor in self.codigos_ibge.items():
            try:
                dic_ibge[key] = tratando_dados_ibge_codigos(
                    codigos=valor, period="all"
                )["Valor"]
            except ValueError:
                print(
                    f"Erro na coleta de dados da variável {key}. Verifique se os códigos {valor} estão ativos em https://sidra.ibge.gov.br/home/pms/brasil."
                )
        if salvar:
            self.salvar_dados(dic_ibge, diretorio, formato)
        else:
            return dic_ibge

    def dados_ibge_link(self, salvar=None, diretorio=None, formato="csv"):
        dic_ibge_link = self.data_index()
        for key, link in self.codigos_ibge_link.items():
            try:
                dic_ibge_link[key] = self.fetch_data_for_code(link, key)
            except ValueError:
                print(
                    f"Erro na coleta da variavel {key}. Verifique se o link esta ativo: {link}"
                )
        if salvar:
            self.salvar_dados(dic_ibge_link, diretorio, formato)
        else:
            return dic_ibge_link

    def dados_ipeadata(self, salvar=None, diretorio=None, formato="csv"):
        dic_ipeadata = self.data_index()
        for nome, codigo in self.codigos_ipeadata.items():
            try:
                dic_ipeadata[nome] = tratatando_dados_ipeadata(
                    codigo_ipeadata={nome: codigo}, data="2000-01-01"
                )
            except ValueError:
                print(
                    f"Erro na coleta da variavel {codigo}. Verifique se o codigo esta ativo: http://www.ipeadata.gov.br/Default.aspx"
                )
        if salvar:
            self.salvar_dados(dic_ipeadata, diretorio, formato)
        else:
            return dic_ipeadata

    def dados_google_trends(
        self, frequencia_datas=None, salvar=None, diretorio=None, formato="csv"
    ):
        if frequencia_datas is None:
            frequencia_datas = "MS"
        dic_google_trends = self.data_index()
        for termo in self.lista_termos_google_trends:
            try:
                dic_google_trends = dic_google_trends.join(
                    tratando_dados_google_trends(
                        [termo],
                        frequencia_data=frequencia_datas,
                        start_date=self.data_inicio,
                    )
                )
            except ValueError:
                print(
                    f"Erro na coleta da variavel {termo}. Verifique se o termo esta ativo: https://trends.google.com/trends/explore?hl=pt-BR"
                )
        if salvar:
            self.salvar_dados(dic_google_trends, diretorio, formato)
        else:
            return dic_google_trends

    def dados_brazil(
        self,
        dados_bcb=True,
        dados_ibge_codigos=True,
        dados_expectativas_inflacao=True,
        dados_ibge_link=True,
        dados_ipeadata=True,
        dados_metas_inflacao=True,
        dados_google_trends=False,
        sem_dados_faltantes=True,
        metodo_preenchimento="ffill",
        salvar=None,
        diretorio=None,
        formato="csv",
    ):

        dados = self.data_index()
        if dados_bcb:
            dados = dados.join(self.dados_banco_central())
        if dados_ibge_codigos:
            dados = dados.join(self.dados_ibge())
        if dados_ibge_link:
            dados = dados.join(self.dados_ibge_link())
        if dados_expectativas_inflacao:
            dados = dados.join(self.dados_expectativas_inflacao())
        if dados_metas_inflacao:
            dados = dados.join(self.dados_metas_inflacao())
        if dados_ipeadata:
            dados = dados.join(self.dados_ipeadata())
        if dados_google_trends:
            dados = dados.join(self.dados_google_trends())
        if sem_dados_faltantes:
            if metodo_preenchimento == "ffill":
                dados = dados.ffill()
                dados = dados.bfill()
        if salvar:
            self.salvar_dados(dados, diretorio, formato)
        else:
            return dados

    def salvar_dados(self, dados, diretorio=None, formato="csv"):
        if not diretorio:
            raise ValueError("Diretório não especificado para salvar o arquivo")
        if formato == "csv":
            dados.to_csv(diretorio)
        elif formato == "excel":
            dados.to_excel(f"{diretorio}.xlsx")
        elif formato == "json":
            dados.to_json(f"{diretorio}.json")
        elif formato == "pickle":
            with open(f"{diretorio}.pkl", "wb") as f:
                pickle.dump(dados, f)
        else:
            raise ValueError("Formato de arquivo não suportado")
