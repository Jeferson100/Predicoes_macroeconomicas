import sys
sys.path.append('..')
from economic_brazil.coleta_dados.economic_data_brazil import EconomicBrazil
from economic_brazil.processando_dados.tratando_dados import TratandoDados
from economic_brazil.treinamento.treinamento_algoritimos import TreinandoModelos
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

SELIC_CODES = {
    "selic": 4189,
    "IPCA-EX2": 27838,
    "IPCA-EX3": 27839,
    "IPCA-MS": 4466,
    "IPCA-MA": 11426,
    "IPCA-EX0": 11427,
    "IPCA-EX1": 16121,
    "IPCA-DP": 16122,}

variaveis_ibge = {
    'ipca': {'codigo': 1737, 'territorial_level': '1', 'ibge_territorial_code': 'all', 'variable': '63'},}

economic_brazil = EconomicBrazil(codigos_banco_central=SELIC_CODES, codigos_ibge=variaveis_ibge, data_inicio="2000-01-01")

dados = economic_brazil.dados_brazil(dados_ipeadata=False)

tratando = TratandoDados(dados)

x_treino, x_teste, y_treino, y_teste,pca, scaler = tratando.tratando_dados()

tuning = TreinandoModelos(x_treino, y_treino, x_teste, y_teste,diretorio='../codigos_rodando/modelos_salvos/modelos_selic/',salvar_modelo=True)

modelos_tunning = tuning.treinar_modelos(redes_neurais=True,sarimax=False)

