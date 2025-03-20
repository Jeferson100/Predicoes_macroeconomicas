import sys
import pandas as pd
from datetime import datetime
import warnings
import pickle
import os
from dotenv import load_dotenv
import time
import asyncio

# sys.path.append("..")
from ..coleta_dados.tratando_economic_brazil_async import (
    tratando_dados_bcb_async,
    tratando_dados_expectativas_async,
    tratando_dados_ibge_link_async,
    tratando_dados_ibge_codigos_async,
    tratatando_dados_ipeadata_async,
    tratando_dados_google_trends_async,
    tratando_dados_ibge_link_producao_agricola_async,
    tratando_dados_ibge_link_colum_brazil_async,
)
from fredapi import Fred
from economic_brazil.coleta_dados.configuracao_apis.api_fred import set_fred_api_key
from pytrends.exceptions import TooManyRequestsError
from typing import List, Dict, Optional, cast

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


class EconomicBrazilAsync:
    def __init__(
        self,
        codigos_banco_central: Optional[Dict[str, int]] = None,
        codigos_ibge: Optional[Dict[str, int]] = None,
        codigos_ibge_link: Optional[Dict[str, str]] = None,
        codigos_ipeadata: Optional[Dict[str, str]] = None,
        codigos_fred: Optional[Dict[str, str]] = None,
        lista_termos_google_trends: Optional[List[str]] = None,
        data_inicio=None,
    ) -> None:
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

    async def fetch_data_for_code_async(self, link: str, column: str) -> pd.DataFrame:
        return await tratando_dados_ibge_link_async(coluna=column, link=link)

    def data_index(self) -> pd.DataFrame:
        data_index = pd.date_range(
            start=self.data_inicio, end=datetime.today().strftime("%Y-%m-%d"), freq="MS"
        )
        return pd.DataFrame(index=data_index)
    
    async def dados_banco_central(
        self,
        salvar: bool = False,
        diretorio: Optional[str] = None,
        formato: str = "csv",
    ) -> pd.DataFrame:
        
        dados = pd.DataFrame()

        async def processar_variavel_banco_central(nome, codigo):
            try:
                # Executa a coleta de dados de forma assíncrona
                resultado = await asyncio.to_thread(
                    tratando_dados_bcb_async,
                    codigo_bcb_tratado={nome: codigo},
                    data_inicio_tratada=self.data_inicio,
                )
                coluna = (await resultado)[nome]

                # Converte para numérico de forma assíncrona, se necessário
                if coluna.dtype == "object":
                    coluna = await asyncio.to_thread(pd.to_numeric, coluna, errors="coerce")

                return nome, coluna
            except ValueError:
                print(
                    f"Erro na coleta de dados da variável {nome}. Verifique se o código {codigo} está ativo https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do?method=prepararTelaLocalizarSeries"
                )
                return nome, pd.Series(dtype="float64")

        # Executa o loop de forma assíncrona para todas as variáveis

        resultados = await asyncio.gather(
            *(processar_variavel_banco_central(nome, codigo) for nome, codigo in self.codigos_banco_central.items())
        )

        # Adiciona os resultados ao DataFrame
        for nome, coluna in resultados:
            dados[nome] = coluna

        # Salva os dados, se necessário
        if salvar:
            self.salvar_dados(dados, diretorio, formato)
        return dados
    
    def dados_banco_central_async(self, salvar: bool = False, diretorio: Optional[str] = None, formato: str = "csv") -> pd.DataFrame:
        """
        Função síncrona que encapsula a chamada assíncrona para dados_banco_central_async.
        """
        return asyncio.run(self.dados_banco_central(salvar=salvar, diretorio=diretorio, formato=formato))
    
    async def dados_expectativas_inflacao(
        self,
        salvar: bool = False,
        diretorio: Optional[str] = None,
        formato: str = "csv",
    ) -> pd.DataFrame:
        """
        Função assíncrona para coletar dados de expectativas de inflação.
        """
        dic_expectativas_inflacao = self.data_index()
        dic_expectativas_inflacao = dic_expectativas_inflacao.join(
           await tratando_dados_expectativas_async()
        )
        if "Mediana" in dic_expectativas_inflacao.columns:
            dic_expectativas_inflacao.rename(
                columns={"Mediana": "ipca_expectativa_focus"}, inplace=True
            )
        if salvar:
            self.salvar_dados(dic_expectativas_inflacao, diretorio, formato)
        return dic_expectativas_inflacao
    
    def dados_expectativas_inflacao_async(
        self,
        salvar: bool = False,
        diretorio: Optional[str] = None,
        formato: str = "csv"
    ) -> pd.DataFrame:
        """
        Função síncrona que encapsula a chamada assíncrona para dados_expectativas_inflacao_async.
        """
        return asyncio.run(self.dados_expectativas_inflacao(
            salvar=salvar,
            diretorio=diretorio,
            formato=formato
        ))
        
    async def dados_ibge(
        self,
        salvar: bool = False,
        diretorio: Optional[str] = None,
        formato: str = "pickle",
    ) -> pd.DataFrame:
        dic_ibge = self.data_index()
        async def processar_variavel_ibge(key, valor):
            try:
                if isinstance(valor, dict):
                    # Executa a coleta de dados de forma assíncrona
                    resultado = await asyncio.to_thread(
                        tratando_dados_ibge_codigos_async,
                        codigos=valor,
                        period="all"
                    )
                    resultado = await resultado
                    return key, resultado["Valor"]
                else:
                    print(f"Tipo de valor não suportado para {key}: {type(valor)}")
                    return key, None
            except ValueError:
                print(
                    f"Erro na coleta de dados da variável {key}. Verifique se os códigos {valor} estão ativos em https://sidra.ibge.gov.br/home/pms/brasil."
                )
                return key, None
        # Executa todas as chamadas em paralelo
        resultados = await asyncio.gather(
            *(processar_variavel_ibge(key, valor) for key, valor in self.codigos_ibge.items())
        )

        # Adiciona os resultados ao DataFrame
        for key, valor in resultados:
            if valor is not None:
                dic_ibge[key] = valor
            
        if salvar:
            self.salvar_dados(dic_ibge, diretorio, formato)
        return dic_ibge
    
    def dados_ibge_async(self, salvar: bool = False, diretorio: Optional[str] = None, formato: str = "pickle") -> pd.DataFrame:
        """
        Função síncrona que encapsula a chamada assíncrona para dados_ibge_async.
        """
        return asyncio.run(self.dados_ibge(salvar=salvar, diretorio=diretorio, formato=formato))
    
    async def dados_ibge_link(
        self,
        salvar: bool = False,
        diretorio: Optional[str] = None,
        formato: str = "csv",
    ) -> pd.DataFrame:
        
        dic_ibge_link = self.data_index()
        
        async def processar_variavel_ibge_link(coluna, link):
            try:
                resultado = await self.fetch_data_for_code_async(link=link, column=coluna)
                return coluna, resultado
            except KeyError:
                # Segunda tentativa: tratando_dados_ibge_link_producao_agricola_async
                try:
                    resultado = await asyncio.to_thread(
                        tratando_dados_ibge_link_producao_agricola_async,
                        link,
                        coluna
                    )
                    resultado = await resultado
                    return coluna, resultado
                except ValueError:
                    print(f"Erro na coleta da variável {coluna}. Verifique se o link está ativo: {link}.")
                    return coluna, None

            except ValueError:
                print(f"Erro na coleta da variável {coluna}. Verifique se o link está ativo: {link}.")
                return coluna, None

        # Executa todas as chamadas em paralelo
        resultados = await asyncio.gather(*(processar_variavel_ibge_link(key, link) for key, link in self.codigos_ibge_link.items())
        )
        
        for key, valor in resultados:
            if valor is not None:
                dic_ibge_link[key] = valor
                # Verifica se precisa usar tratando_dados_ibge_link_colum_brazil
                if key not in dic_ibge_link.columns or bool(dic_ibge_link[key].isnull().all()):
                    try:
                        resultado = await asyncio.to_thread(
                            tratando_dados_ibge_link_colum_brazil_async,
                            key,
                            self.codigos_ibge_link[key]
                        )
                        resultado = await resultado
                        dic_ibge_link[key] = resultado
                    except ValueError:
                        print(f"Erro na coleta da variável {key}. Verifique se o link está ativo: {self.codigos_ibge_link[key]}.")
        if salvar:
            await asyncio.to_thread(self.salvar_dados, dic_ibge_link, diretorio, formato)
        return dic_ibge_link

    def dados_ibge_link_async(self, salvar: bool = False, diretorio: Optional[str] = None, formato: str = "csv"):
        """
        Função síncrona que encapsula a chamada assíncrona para dados_ibge_link_async.
        """
        return asyncio.run(self.dados_ibge_link(salvar=salvar, diretorio=diretorio, formato=formato))

    
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
    
    
    
if __name__ == "__main__":
    start = time.time()
    load_dotenv()
    economic_brazil = EconomicBrazilAsync()
    print('Index:')
    print(economic_brazil.data_index())
    
    print('fetch_data_for_code_async:')
    loop = asyncio.get_event_loop()
    print(loop.run_until_complete(economic_brazil.fetch_data_for_code_async(indicadores_ibge_link_padrao['pib'], 'PIB')))
    
    print('Banco Central:')
    print(economic_brazil.dados_banco_central_async())
    
    print("Expectativas Inflação:")
    expectativas_inflacao_data = economic_brazil.dados_expectativas_inflacao_async(
        salvar=True, diretorio="dados_expectativas_inflacao.csv"
    )
    print(expectativas_inflacao_data)
    
    print("IBGE:")
    ibge_data = economic_brazil.dados_ibge_async(salvar=True, diretorio="dados_ibge")
    print(ibge_data)
    
    print("IBGE Link:")
    ibge_link_data = economic_brazil.dados_ibge_link_async(salvar=True, diretorio="dados_ibge_link")
    print(ibge_link_data)
    
    print('*' * 100)
    end = time.time()
    print(f"Tempo de execução: {end - start} segundos")
    #loop.close()
    