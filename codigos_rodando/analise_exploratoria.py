import sys
sys.path.append('..')
from economic_brazil.coleta_dados.economic_data_brazil import data_economic
from economic_brazil.visualizacoes_graficas.codigos_graficos import Graficos
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

#Coleta
dados = data_economic()
#info
dados.info().to_csv("/workspaces/Predicoes_macroeconomicas/imagens/info.csv")
#describe
dados.describe().to_csv("/workspaces/Predicoes_macroeconomicas/imagens/describe.csv")
#graficos
graficos = Graficos()
#
#graficos.go_plotar(dados, save=True, diretorio="dados/graficos.png")

graficos.plotar_temporal(dados, save=True, diretorio="/workspaces/Predicoes_macroeconomicas/imagens")

graficos.plotar_heatmap(dados, save=True, diretorio="/workspaces/Predicoes_macroeconomicas/imagens/graficos_heatmap.png")



