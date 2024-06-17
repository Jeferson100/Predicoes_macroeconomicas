import sys
sys.path.append('..')
from economic_brazil.coleta_dados.economic_data_brazil import EconomicBrazil
#from economic_brazil.coleta_dados.tratando_economic_brazil import tratando_dados_bcb, tratando_dados_ibge_link, tratando_metas_inflacao, tratando_dados_expectativas, tratando_dados_ibge_codigos
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

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
    'seguro desemprego'
]

DATA_INICIO = "2000-01-01"

dados = EconomicBrazil(codigos_banco_central=SELIC_CODES, codigos_ibge=variaveis_ibge, codigos_ibge_link=variaveis_ibge, codigos_ipeadata=codigos_ipeadata_padrao, lista_termos_google_trends=lista_google_trends, data_inicio=DATA_INICIO)

dados.dados_brazil(salvar=True, diretorio='economic_data_brazil', formato='pickle')

print('Coletando dados Brazil')

dados.dados_ibge(salvar=True, diretorio='dados_ibge', formato='pickle')

print('Coletando dados IBGE')

dados.dados_banco_central(salvar=True, diretorio='dados_banco_central', formato='pickle')

print('Coletando dados Banco Central')

dados.dados_expectativas_inflacao(salvar=True, diretorio='dados_expectativas_inflacao', formato='pickle')

print('Coletando dados Expectativas Inflacao')

dados.dados_metas_inflacao(salvar=True, diretorio='dados_metas_inflacao', formato='pickle')

print('Coletando dados Metas Inflacao')

dados.dados_ibge_link(salvar=True, diretorio='dados_ibge_link', formato='pickle')

print('Coleta terminanda com sucesso!')





