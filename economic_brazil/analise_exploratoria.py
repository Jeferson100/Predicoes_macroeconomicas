from .economic_data_brazil import data_economic
from .codigos_graficos import Graficos

import warnings

warnings.filterwarnings("ignore")

dados = data_economic()
dados.info()
dados.sumary()
graficos = Graficos()
