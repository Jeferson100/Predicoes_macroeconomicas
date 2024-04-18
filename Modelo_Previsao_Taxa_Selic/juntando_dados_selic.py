import pandas as pd
import numpy as np
from datetime import date, datetime
from tratando_dados_selic import tratando_dados_bcb, tratando_dados_ibge_codigos, tratando_dados_expectativas, tratando_dados_ibge_link, tratando_metas_inflacao

data_inicio = '2000-01-01'

data_index = pd.date_range(start=data_inicio,end=datetime.today().strftime('%Y-%m-%d'),freq='MS')

dados = pd.DataFrame(index=data_index)

dados = tratando_dados_bcb(data_inicio_tratada=data_inicio)

dados['expectativas'] = tratando_dados_expectativas()

dados['diferenca_meta_efetiva'] = tratando_metas_inflacao()['diferenca_meta_efetiva']


