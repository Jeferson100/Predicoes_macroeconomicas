import pandas as pd
import numpy as np
from datetime import date
from bcb import sgs
import sidrapy
from bcb import Expectativas

#Dados BCB
selic = {'selic':4189,'IPCA-EX2':27838,'IPCA-EX3':27839,
         'IPCA-MS':4466,'IPCA-MA':11426,'IPCA-EX0':11427,
        'IPCA-EX1':16121,'IPCA-DP':16122}

def dados_bcb(codigo_bcb=selic,data_inicio='2000-01-01'):
    dados = sgs.get(codigo_bcb,start = data_inicio)
    return dados


#DADOS IBGE
def dados_ibge_codigos(codigo='1737',territorial_level='1',ibge_territorial_code='all',variable='63',period='all'):
    ipca = sidrapy.get_table(table_code = codigo,
                             territorial_level = territorial_level,
                             ibge_territorial_code = ibge_territorial_code,
                             variable = variable,
                             period = period)
    return ipca

def dados_ibge_link(cabecalho=3, url='https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93405/d/v6561%201/l/v,p%2Bc11255,t'):
    # carregar a tabela em um DataFrame
    dados_link = pd.read_excel(url, header=cabecalho)
    return dados_link


# Dados Expectativas/Focus
def dados_expectativas_focus(indicador='IPCA', tipo_expectativa='ExpectativaMercadoMensais', data_inicio='2000-01-01'):
    # End point
    em = Expectativas()
    ep = em.get_endpoint(tipo_expectativa)

    # Dados do IPCA

    ipca_expec = (ep.query()
    .filter(ep.Indicador == indicador)
    .filter(ep.Data >= data_inicio)
    .filter(ep.baseCalculo == 0)
    .select(ep.Indicador, ep.Data, ep.Media, ep.Mediana, ep.DataReferencia,ep.baseCalculo)
    .collect()
    )
    return ipca_expec


print(dados_expectativas_focus())
print(dados_bcb())
print(dados_ibge_codigos())
print(dados_ibge_link())