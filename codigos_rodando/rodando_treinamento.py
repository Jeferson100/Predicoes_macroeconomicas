import sys
sys.path.append('..')
from economic_brazil.coleta_dados.economic_data_brazil import data_economic
from economic_brazil.processando_dados.tratando_dados import TratandoDados
from economic_brazil.treinamento.treinamento_algoritimos import TreinandoModelos
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

dados = data_economic()

tratando = TratandoDados(dados)
x_treino, x_teste, y_treino, y_teste,pca, scaler = tratando.tratando_dados()

tuning = TreinandoModelos(x_treino, y_treino, x_teste, y_teste,diretorio='../codigos_rodando/modelos_salvos/',salvar_modelo=True)

modelos_tunning = tuning.treinar_modelos(redes_neurais=True,sarimax=True)