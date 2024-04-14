from dados_selic import dados_bcb, dados_ibge, dados_expectativas_focus
import pandas as pd
import numpy as np
from datetime import date

def tratando_dados_expectativas():
    ipca_expec = dados_expectativas_focus()
    dados_ipca = ipca_expec.copy()
    dados_ipca = dados_ipca[::-1]
    dados_ipca['monthyear'] = pd.to_datetime(dados_ipca['Data']).apply(lambda x: x.strftime("%Y-%m"))
    dados_ipca = dados_ipca.groupby('monthyear').mean()
    # criar índice com o formato "YYYY-MM"
    dados_ipca.index = pd.to_datetime(dados_ipca.index, format='%Y-%m')

    # adicionar o dia como "01"
    dados_ipca.index = dados_ipca.index.to_period('M').to_timestamp()

    return dados_ipca
