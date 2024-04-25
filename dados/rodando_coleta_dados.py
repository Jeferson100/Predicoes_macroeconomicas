import sys
sys.path.append('..')
from economic_brazil.coleta_dados.economic_data_brazil import data_economic
from economic_brazil.coleta_dados.tratando_economic_brazil import tratando_dados_bcb, tratando_dados_ibge_link, tratando_metas_inflacao, tratando_dados_expectativas, tratando_dados_ibge_codigos
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

data_economic(salvar=True, formato="csv",diretorio="economic_data_brazil.csv")

tratando_dados_bcb(salvar=True, formato="csv",diretorio="dados_bcb.csv")

tratando_dados_expectativas(salvar=True, formato="csv",diretorio="dados_expectativas.csv")

tratando_metas_inflacao(salvar=True, formato="csv",diretorio="dados_metas_inflacao.csv")

tratando_dados_ibge_codigos(salvar=True, formato="csv",diretorio="dados_ibge_codigos.csv")

indicadores = {
            "pib": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/90707/d/v6561%201/l/v,p%2Bc11255,t",
            "despesas_publica": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93405/d/v6561%201/l/v,p%2Bc11255,t",
            "capital_fixo": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93406/d/v6561%201/l/v,p%2Bc11255,t",
            "producao_industrial_manufatureira": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela8158.xlsx&terr=N&rank=-&query=t/8158/n1/all/v/11599/p/all/c543/129278/d/v11599%205/l/v,p%2Bc543,t",
        }
for indicador, url in indicadores.items():
    tratando_dados_ibge_link(indicador, url, salvar=True, formato="csv",diretorio=f"dados_ibge_link_{indicador}.csv")


print('Pronto Terminou, dados coletados!')


