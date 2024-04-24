import sys
sys.path.append('..')
from economic_brazil.economic_data_brazil import data_economic
from economic_brazil.codigos_graficos import Graficos
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

dados = data_economic()
dados.info()
dados.sumary()
graficos = Graficos()