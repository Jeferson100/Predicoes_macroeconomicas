import sys
sys.path.append('..')
from economic_brazil.economic_data_brazil import data_economic
from economic_brazil.tratando_economic_brazil import tratando_dados_bcb, tratando_dados_ibge_link, tratando_metas_inflacao, tratando_dados_expectativas, tratando_dados_ibge_codigos
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

data_economic(salvar=True, formato="csv",diretorio="dados/economic_data_brazil.csv")



