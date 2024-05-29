import sys
sys.path.append('..')
from economic_brazil.coleta_dados.economic_data_brazil import data_economic
from economic_brazil.processando_dados.tratando_dados import TratandoDados
from economic_brazil.treinamento.treinamento_algoritimos import carregar
from economic_brazil.analisando_modelos.analise_modelos_regressao import MetricasModelosDicionario,PredicaosModelos
from economic_brazil.analisando_modelos.regressao_conformal import ConformalRegressionPlotter
from economic_brazil.predicao_valores_futuros import KerasTrainedRegressor, Predicao
import pandas as pd
from joblib import parallel_config
import warnings
import pickle
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
    diretorio='/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/predicao_treino.png',
)

metri.plotando_predicoes(
    y_teste,
    predicoes_teste,
    index=index_teste,
    title="Predições nos dados de teste",
    save=True,
    diretorio='/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/predicao_teste.png',
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

##melhor modelo baseado em MENOR erro absoluto (MAE)
melhor_modelo = metrica_teste['MAE'].idxmin()
print(f'Melhor modelo baseado na MAE mais baixo, com valor de {metrica_teste["MAE"].min()}:',melhor_modelo)

#Conformal
index_treino_conformal = dados[dados.index <= data_divisao_treino_teste].index
index_teste_conformal = dados[dados.index > data_divisao_treino_teste].index

with parallel_config(backend='threading', n_jobs=2):
    if melhor_modelo == 'redes_neurais':
        conformal = ConformalRegressionPlotter(KerasTrainedRegressor(modelos_carregados[melhor_modelo]), x_treino_recorrente, x_teste_recorrente, y_treino[1:], y_teste[1:])
        y_pred, y_pis, _,_ = conformal.regressao_conformal()
        conformal.plot_prediction_intervals(index_train=index_treino_conformal, index_test=index_teste_conformal,title=f'Predição Intervals {melhor_modelo}',save=True,diretorio='/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/regressao_conforma_teste.png')
    else:
        conformal = ConformalRegressionPlotter(modelos_carregados[melhor_modelo], x_treino, x_teste, y_treino[1:], y_teste[1:])
        y_pred, y_pis, _,_ = conformal.regressao_conformal()
        conformal.plot_prediction_intervals(index_train=index_treino_conformal, index_test=index_teste_conformal,title=f'Predição Intervals {melhor_modelo}',save=True,diretorio='/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/regressao_conforma_teste.png')

dados_corformal = {
    'data': index_teste[-1],
    'predicao': y_pred,
    'intervalo_lower': y_pis.squeeze()[0:,0],
    'intervalo_upper': y_pis.squeeze()[0:,1],}

pd.DataFrame(dados_corformal).to_csv('/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/regressao_conforma_teste.csv',index=False)
    
## Predicao futuro
with parallel_config(backend='threading', n_jobs=2):
    predicao = Predicao(x_treino,y_treino,tratando,dados,melhor_modelo,modelos_carregados[melhor_modelo],coluna='selic')
dados_predicao_futuro, dados_futuro, index_futuro = predicao.criando_dados_futuros()
dados_predicao = predicao.criando_dataframe_predicoes()
print(dados_predicao)
data_predicao,intervalo_lower,intervalo_upper,predicao_proximo_mes = predicao.predicao_ultimo_periodo()
with open('/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/dados_futuro.pkl', 'wb') as pickle_file:
    pickle.dump(dados_futuro, pickle_file)
print(f'Data da predição:{data_predicao}, Valor da predição:{predicao_proximo_mes}, Intervalo de predição [lower,upper] :{intervalo_lower,intervalo_upper}')
predicao.plotando_predicoes(save=True,diretorio='/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/predicao_futuro.png')