from .economic_data_brazil import data_economic
from .codigos_graficos import plotar
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

dados = data_economic()

plotar(dados)

dados.info()

dados.sumary()


