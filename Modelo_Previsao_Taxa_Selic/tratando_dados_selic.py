from dados_selic import dados_bcb, dados_ibge_codigos, dados_expectativas_focus, dados_ibge_link
import pandas as pd
import numpy as np
from datetime import date, datetime



def tratando_dados_expectativas():
    ipca_expec = dados_expectativas_focus()
    dados_ipca = ipca_expec.copy()
    dados_ipca = dados_ipca[::-1]
    dados_ipca['monthyear'] = pd.to_datetime(dados_ipca['Data']).apply(lambda x: x.strftime("%Y-%m"))
    dados_ipca = dados_ipca.groupby('monthyear')['Mediana'].mean()
    # criar índice com o formato "YYYY-MM"
    dados_ipca.index = pd.to_datetime(dados_ipca.index, format='%Y-%m')

    # adicionar o dia como "01"
    dados_ipca.index = dados_ipca.index.to_period('M').to_timestamp()

    return dados_ipca

#Tratando dados IBGE/SIDRAPY

def trimestral_para_mensal(df):
  """
   A função recebe um DataFrame df com valores trimestrais do PIB. Primeiro, ela aplica a interpolação para obter os valores mensais, usando o método resample com uma frequência de 'M' e o método interpolate para preencher os valores faltantes. 
   Em seguida, ela percorre os valores de cada trimestre e distribui a variação trimestral em cada um dos três meses dentro do trimestre, adicionando um terço do valor trimestral aos dois meses intermediários. Finalmente, ela retorna o DataFrame 
   interpolado e transformado em mensal.
   """
  
  # Primeiro, definimos uma nova frequência mensal e aplicamos a interpolação
  df_mensal = df.resample('MS').interpolate()
    
  # Em seguida, distribuímos a variação trimestral em cada um dos três meses dentro do trimestre
  for i in range(1, len(df)):
    if i % 3 != 0:
        val = df.iloc[i].values[0]
        month_val = val / 3
        df_mensal.iloc[i*2-1] += month_val
        df_mensal.iloc[i*2] += month_val
    
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
    lista_trimestre.append(dados.index[i][-4:]+'-'+'0'+dados.index[i][0])
  return lista_trimestre

def transforma_para_mes_incial_trimestre(dados):
  lista_mes = []
  for i in range(len(dados.index)):
    trimestre = dados.index.month[i]
    ano = str(dados.index.year[i])
    lista_mes.append(str(np.where(trimestre == 1,ano+'-'+'0'+str(trimestre),
        np.where(trimestre == 2,ano+'-'+'0'+str(trimestre+2),
        np.where(trimestre==3,ano+'-'+'0'+str(trimestre+4),
        np.where(trimestre==4,ano+'-'+str(trimestre+6),0))))))
  return lista_mes

###Tratando dados IBGE

def tratando_dados_ibge_codigos():
    ibge_codigos = dados_ibge_codigos()
    ibge_codigos.columns = ibge_codigos.iloc[0,:]
    ibge_codigos = ibge_codigos.iloc[1:,:]
    ibge_codigos['data'] = ibge_codigos['Mês (Código)'].apply(converter_mes_para_data)
    ibge_codigos.index = ibge_codigos['data']
    ibge_codigos['Valor'] = ibge_codigos['Valor'][1:].astype(float)
    return ibge_codigos


def tratando_dados_ibge_link(coluna='pib',link='https://sidra.ibge.gov.br/estatisticas/sociais/indicadores-geograficos/6579/taxa-de-selic-ao-ano'):
    dado_ibge= dados_ibge_link(link=link)
    ibge_link = dado_ibge.T
    ibge_link = ibge_link[[1]]
    ibge_link.columns = coluna
    ibge_link = ibge_link.astype(float)
    ibge_link.index = pd.to_datetime(trimestre_string_int(ibge_link))
    ibge_link.index = pd.to_datetime(transforma_para_mes_incial_trimestre(ibge_link))
    ibge_link = ibge_link.resample('MS').fillna(method='ffill')
    return ibge_link


###Tratando dados BCB

def tratando_dados_bcb():
   return


###Tratando dados Expectativas

def tratando_dados_expectativas():
    ipca_expec = dados_expectativas_focus()
    dados_ipca = ipca_expec.copy()
    dados_ipca = dados_ipca[::-1]
    dados_ipca['monthyear'] = pd.to_datetime(dados_ipca['Data']).apply(lambda x: x.strftime("%Y-%m"))
    dados_ipca = dados_ipca.groupby('monthyear')['Mediana'].mean()
    # criar índice com o formato "YYYY-MM"
    dados_ipca.index = pd.to_datetime(dados_ipca.index, format='%Y-%m')

    # adicionar o dia como "01"
    dados_ipca.index = dados_ipca.index.to_period('M').to_timestamp()

    return dados_ipca

print(tratando_dados_ibge_codigos())
print(tratando_dados_expectativas())
print('PIB TRIMESTRAL:')
print(tratando_dados_ibge_link(coluna='pib',link='https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/90707/d/v6561%201/l/v,p%2Bc11255,t'))
print('Despesa de consumo da administracao publica(TRIMESTRAL)')
print(tratando_dados_ibge_link(coluna='despesa',link='https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93405/d/v6561%201/l/v,p%2Bc11255,t'))
print('Formacao bruta de capital fixo')
print(tratando_dados_ibge_link(coluna='capital',link='https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93406/d/v6561%201/l/v,p%2Bc11255,t'))
print('Produção indústrias de manufatureiras')
print(tratando_dados_ibge_link(coluna='producao',link='https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela8158.xlsx&terr=N&rank=-&query=t/8158/n1/all/v/11599/p/all/c543/129278/d/v11599%205/l/v,p%2Bc543,t'))



