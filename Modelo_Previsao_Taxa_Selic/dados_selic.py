import pandas as pd
import numpy as np
from datetime import date
from bcb import sgs
import sidrapy
from bcb import Expectativas

## melhore a funcao dados_bcb onde vc pode passar o codigo da variavel

def dados_bcb(codigo_bcb:dict,data_inicio):
    dados = sgs.get(codigo_bcb,start = data_inicio)
    return dados

selic = {'selic':4189,'IPCA-EX2':27838,'IPCA-EX3':27839,
         'IPCA-MS':4466,'IPCA-MA':11426,'IPCA-EX0':11427,
        'IPCA-EX1':16121,'IPCA-DP':16122}

inicio = '2000-01-01'
dados = dados_bcb(selic,inicio)
print(dados)

def dados_ibge(codigo_ibge):
    return

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