import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import matplotlib
matplotlib.rcParams['figure.figsize'] = (16,8)


def consulta_bc(codigo_bcb):
 """
-Essa função vai no site do banco central(https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do?method=prepararTelaLocalizarSeries) e coleta um histórico de indicadores econômicos.
-Para fazer a seleção só é necessario indicar o código da mesma.


  """






 url='http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json'.format(codigo_bcb)
 df = pd.read_json(url)
 df['data'] = pd.to_datetime(df['data'], dayfirst=True)
 df.set_index('data', inplace=True)
 return df