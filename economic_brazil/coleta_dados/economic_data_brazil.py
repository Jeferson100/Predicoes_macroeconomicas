#!/usr/bin/env python
import sys
import pandas as pd
from datetime import datetime
import warnings
import pickle
import os
from dotenv import load_dotenv

sys.path.append("..")
from economic_brazil.coleta_dados.tratando_economic_brazil import (
    tratando_dados_bcb,
    tratando_dados_expectativas,
    tratando_dados_ibge_link,
    tratando_dados_ibge_codigos,
    tratatando_dados_ipeadata,
    tratando_dados_google_trends,
    tratando_dados_ibge_link_producao_agricola,
    tratando_dados_ibge_link_colum_brazil,
)
from fredapi import Fred
from economic_brazil.coleta_dados.configuracao_apis.api_fred import set_fred_api_key
from pytrends.exceptions import TooManyRequestsError
import asyncio

warnings.filterwarnings("ignore")

DATA_INICIO = "2000-01-01"

variaveis_banco_central_padrao = {
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

variaveis_ibge_padrao = {
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

indicadores_ibge_link_padrao = {
    "pib": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6564/p/all/c11255/90707/d/v6564%201/l/v,p,t%2Bc11255&verUFs=false&verComplementos2=false&verComplementos1=false&omitirIndentacao=false&abreviarRotulos=false&exibirNotas=false&agruparNoCabecalho=false",
    "despesas_publica": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93405/d/v6561%201/l/v,p%2Bc11255,t",
    "capital_fixo": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93406/d/v6561%201/l/v,p%2Bc11255,t",
    "producao_industrial_manufatureira": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela8158.xlsx&terr=N&rank=-&query=t/8158/n1/all/v/11599/p/all/c543/129278/d/v11599%205/l/v,p%2Bc543,t",
    "soja": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela6588.xlsx&terr=N&rank=-&query=t/6588/n1/all/v/35/p/all/c48/0,39443/l/v,p%2Bc48,t",
    "milho_1": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela6588.xlsx&terr=N&rank=-&query=t/6588/n1/all/v/35/p/all/c48/0,39441/l/v,p%2Bc48,t",
    "milho_2": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela6588.xlsx&terr=N&rank=-&query=t/6588/n1/all/v/35/p/all/c48/0,39442/l/v,p%2Bc48,t",
}

lista_google_trends_padrao = [
    "seguro desemprego",
]

codigos_fred_padrao = {
    "nasdaq100": "NASDAQ100",
    "taxa_cambio_efetiva": "RBBRBIS",
    "cboe_nasdaq": "VXNCLS",
    "taxa_juros_interbancaria": "IRSTCI01BRM156N",
    "atividade_economica_eua": "USPHCI",
    "indice_confianca_manufatura": "BSCICP03BRM665S",
    "indice_confianca_exportadores": "BSXRLV02BRM086S",
    "indice_tendencia_emprego": "BRABREMFT02STSAM",
    "indice_confianca_consumidor": "CSCICP03BRM665S",
    "capacidade_instalada": "BSCURT02BRM160S",
}


class EconomicBrazil:
    def __init__(
        self,
        codigos_banco_central=None,
        codigos_ibge=None,
        codigos_ibge_link=None,
        codigos_ipeadata=None,
        codigos_fred=None,
        lista_termos_google_trends=None,
        data_inicio=None,
    ):
        self.codigos_banco_central = (
            codigos_banco_central or variaveis_banco_central_padrao
        )
        self.codigos_ibge = codigos_ibge or variaveis_ibge_padrao
        self.codigos_ibge_link = codigos_ibge_link or indicadores_ibge_link_padrao
        self.codigos_ipeadata = codigos_ipeadata or codigos_ipeadata_padrao
        self.lista_termos_google_trends = (
            lista_termos_google_trends or lista_google_trends_padrao
        )
        self.codigos_fred = codigos_fred or codigos_fred_padrao
        self.data_inicio = data_inicio or DATA_INICIO

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

            except KeyError:
                dic_ibge_link[key] = tratando_dados_ibge_link_producao_agricola(
                    link, key
                )

            except ValueError:
                print(
                    f"Erro na coleta da variável {key}. Verifique se o link está ativo: {link}."
                )
            try:
                if (
                    key not in dic_ibge_link.columns
                    or dic_ibge_link[key].isnull().all()
                ):
                    dic_ibge_link[key] = tratando_dados_ibge_link_colum_brazil(
                        key, link
                    )
            except ValueError:
                print(
                    f"Erro na coleta da variável {key}. Verifique se o link está ativo: {link}."
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
            except KeyError:
                print(
                    f"Erro na coleta da variavel {codigo}. Verifique se o codigo esta ativo: http://www.ipeadata.gov.br/Default.aspx"
                )
        try:
            if (
                "caged_antigo" in dic_ipeadata.columns
                and "caged_novo" in dic_ipeadata.columns
            ):
                dic_ipeadata["caged_junto"] = pd.concat(
                    [
                        dic_ipeadata.caged_antigo.dropna(),
                        dic_ipeadata.caged_novo.dropna(),
                    ]
                )
                dic_ipeadata = dic_ipeadata.drop(["caged_antigo", "caged_novo"], axis=1)
        except ValueError:
            print(
                "Erro na juncao da variavel caged antigo e novo. Verifique se o codigo esta ativo: http://www.ipeadata.gov.br/Default.aspx"
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
            except TooManyRequestsError:
                print(
                    f"Too many requests error for {termo}. Skipping this term for now."
                )

        if salvar:
            self.salvar_dados(dic_google_trends, diretorio, formato)
        else:
            return dic_google_trends

    def dados_fred(self, salvar=None, diretorio=None, formato="csv"):
        dic_fred = self.data_index()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.abspath(os.path.join(base_dir, ".env"))
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            set_fred_api_key()
            sys.exit(
                "Chave de API do FRED salva com sucesso. Encerrando o script. Rode o script novamente para coletar os dados."
            )
        fred = Fred(api_key=api_key)
        if fred:
            for key, codes in self.codigos_fred.items():
                try:
                    dic_fred[key] = fred.get_series(codes)
                except ValueError:
                    print(
                        f"Erro na coleta da variável {key}. Verifique se os códigos {codes} estão ativos em https://fred.stlouisfed.org/."
                    )
        else:
            print(
                "Vefique se a chave da API esta definida corretamente em https://fred.stlouisfed.org/."
            )

        if salvar:
            self.salvar_dados(dic_fred, diretorio, formato)
        else:
            return dic_fred

    def dados_brazil(
        self,
        dados_bcb=True,
        dados_ibge_codigos=True,
        dados_expectativas_inflacao=True,
        dados_ibge_link=True,
        dados_ipeadata=True,
        dados_google_trends=False,
        dados_fred=False,
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
        if dados_ipeadata:
            dados = dados.join(self.dados_ipeadata())
        if dados_google_trends:
            dados = dados.join(self.dados_google_trends())
        if dados_fred:
            dados = dados.join(self.dados_fred())
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
