import sys
sys.path.append('..')
from economic_brazil.economic_data_brazil import data_economic
from economic_brazil.codigos_graficos import Graficos
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

#Coleta
dados = data_economic()
#info
dados.info()
#describe
dados.describe()
#graficos
graficos = Graficos()
#
graficos.go_plotar(dados)

graficos.plotar_temporal(dados)



