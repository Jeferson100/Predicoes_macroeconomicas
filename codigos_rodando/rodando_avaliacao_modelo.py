import sys
sys.path.append('..')
from economic_brazil.coleta_dados.economic_data_brazil import data_economic
from economic_brazil.processando_dados.tratando_dados import TratandoDados
from economic_brazil.treinamento.treinamento_algoritimos import carregar
from economic_brazil.analisando_modelos.analise_modelos_regressao import MetricasModelosDicionario,PredicaosModelos
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Coleta
dados = data_economic()

#Tratando
tratando = TratandoDados(dados)
x_treino, x_teste, y_treino, y_teste,pca, scaler = tratando.tratando_dados()
data_divisao_treino_teste = tratando.data_divisao_treino_teste()

#Carregando modelos
modelos_carregados = carregar(diretorio='../codigos_rodando/modelos_salvos/',gradiente_boosting=True, xg_boost=True, cat_boost=True, regressao_linear=True, redes_neurais=True, sarimax=True)

#Prevendo os dados de treino e teste
predi = PredicaosModelos(modelos_carregados, x_treino, y_treino, x_teste, y_teste)
predicoes_treino, predicoes_teste = predi.predicoes()
x_treino, y_treino, x_teste, y_teste, x_treino_recorrente, y_treino_recorrente, x_teste_recorrente, y_teste_recorrente = predi.return_dados()

#Calculando metricas
metri = MetricasModelosDicionario()
metrica_teste = metri.calculando_metricas(predicoes_teste, y_teste, y_teste_recorrente)
metrica_treino = metri.calculando_metricas(predicoes_treino, y_treino, y_treino_recorrente)

metrica_teste
metrica_treino

metrica_teste.to_csv("/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/metricas_teste.csv")
metrica_treino.to_csv("/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/metricas_treino.csv")

index_treino = dados[dados.index <= data_divisao_treino_teste][6:].index
index_teste = dados[dados.index > data_divisao_treino_teste][4:].index

# Plotando predicoes
metri.plotando_predicoes(
    y_treino,
    predicoes_treino,
    index=index_treino,
    title="Predições nos dados de treino",
    save=True,
    diretorio='/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/'
)

metri.plotando_predicoes(
    y_teste,
    predicoes_teste,
    index=index_teste,
    title="Predições nos dados de teste",
)

metri.plotando_predicoes_go_treino_teste(
    y_treino,
    y_teste,
    predicoes_treino,
    predicoes_teste,
    index_treino,
    index_teste,
    save=True,
    diretorio='/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/predicao_treino_teste.png',
    type_arquivo='png'
)