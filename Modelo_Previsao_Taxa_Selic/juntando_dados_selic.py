import pandas as pd
import numpy as np
from datetime import date, datetime
from tratando_dados_selic import tratando_dados_bcb, tratando_dados_ibge_codigos, tratando_dados_expectativas, tratando_dados_ibge_link, tratando_metas_inflacao
import warnings

warnings.filterwarnings('ignore')

data_inicio = '2000-01-01'

selic = {'selic':4189,'IPCA-EX2':27838,'IPCA-EX3':27839,
         'IPCA-MS':4466,'IPCA-MA':11426,'IPCA-EX0':11427,
        'IPCA-EX1':16121,'IPCA-DP':16122}
def dados_juntos_selic(codigos_banco_central=selic,data_inicio='2000-01-01',expectativas_inflacao=True,meta_inflacao=True,banco_central=True,pib=True,despesas_publica=True,
                       capital_fixo=True,producao_industrial_manufatureira=True,ipca=True):
   
    data_index = pd.date_range(start=data_inicio,end=datetime.today().strftime('%Y-%m-%d'),freq='MS')
    dados = pd.DataFrame(index=data_index)
    if banco_central:
        if not isinstance(codigos_banco_central, dict):
            print("Código BCB deve ser um dicionário. Usando valor padrão.")
        codigos_banco_central = selic
        dados = tratando_dados_bcb(codigo_bcb_tratado=codigos_banco_central, data_inicio_tratada=data_inicio)
    if expectativas_inflacao:
        dados['expectativas_inflacao'] = tratando_dados_expectativas()   
    if meta_inflacao:
        dados['diferenca_meta_efetiva'] = tratando_metas_inflacao()['diferenca_meta_efetiva']
    if pib:
        dados['pib'] = tratando_dados_ibge_link(coluna='pib',link='https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/90707/d/v6561%201/l/v,p%2Bc11255,t')
    if despesas_publica:
        dados['despesas_publicas'] = tratando_dados_ibge_link(coluna='despesas_publicas',link='https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93405/d/v6561%201/l/v,p%2Bc11255,t')  
    if capital_fixo:
        dados['capital_fixo'] = tratando_dados_ibge_link(coluna='despesas_publicas',link='https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93406/d/v6561%201/l/v,p%2Bc11255,t')
    if producao_industrial_manufatureira:
        dados['producao_ind_manufatureira'] = tratando_dados_ibge_link(coluna='producao',link='https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela8158.xlsx&terr=N&rank=-&query=t/8158/n1/all/v/11599/p/all/c543/129278/d/v11599%205/l/v,p%2Bc543,t')
    if ipca:
        dados['ipca'] = tratando_dados_ibge_codigos()['Valor']
    return dados
        
print(dados_juntos_selic())


