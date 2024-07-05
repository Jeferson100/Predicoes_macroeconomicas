import sys
sys.path.append('..')
from economic_brazil.treinamento.treinamento_algoritimos import carregar
from economic_brazil.analisando_modelos.analise_modelos_regressao import MetricasModelosDicionario,PredicaosModelos
from economic_brazil.analisando_modelos.regressao_conformal import ConformalRegressionPlotter
from economic_brazil.predicao_valores_futuros import KerasTrainedRegressor, Predicao
import pandas as pd
from joblib import parallel_config
import warnings
import pickle
import os
warnings.filterwarnings("ignore", category=UserWarning)
path_codigos_rodando = os.path.join(os.getcwd())

##########################################################################################################DEFININDO VARIAVEIS############################################################################################################

variavel_predicao = 'taxa_desocupacao'

##########################################################################################CARREGANDO DADOS############################################################################################################

arquivo_pib = path_codigos_rodando+f'/avaliacao_modelos/apresentacao_streamlit/dados_treinamento_{variavel_predicao}.pkl'
dados_carregados = pickle.load(open(arquivo_pib, 'rb'))

dados = dados_carregados['dados']
x_treino = dados_carregados['x_treino']
x_teste = dados_carregados['x_teste']
y_treino = dados_carregados['y_treino']
y_teste = dados_carregados['y_teste']
if 'pca' in dados_carregados.keys():
    pca = dados_carregados['pca']
if 'scaler' in dados_carregados.keys():    
    scaler = dados_carregados['scaler']
if 'rfe_model' in dados_carregados.keys():
    rfe_model = dados_carregados['rfe_model']
if 'smart_model' in dados_carregados.keys():
    smart_model = dados_carregados['smart_model']
if 'variance_model' in dados_carregados.keys():
    variance_model = dados_carregados['variance_model']
data_divisao_treino_teste = dados_carregados['data_divisao_treino_teste']
tratando = dados_carregados['tratando']

##########################################################################################################CARREGANDO MODELOS############################################################################################################

modelos_carregados = carregar(diretorio=path_codigos_rodando+f'/modelos_salvos/modelos_{variavel_predicao}/',gradiente_boosting=True, xg_boost=True, cat_boost=True, regressao_linear=True, redes_neurais=True, sarimax=False)


##########################################################################################################PREVENDO OS DADOS DE TREINO E TESTE################################################################################
predi = PredicaosModelos(modelos_carregados, x_treino, y_treino, x_teste, y_teste)
predicoes_treino, predicoes_teste = predi.predicoes()
x_treino, y_treino, x_teste, y_teste, x_treino_recorrente, y_treino_recorrente, x_teste_recorrente, y_teste_recorrente = predi.return_dados()

###########################################################################################################################CALCULANDO METRICAS################################################################################
metri = MetricasModelosDicionario()
metrica_teste = metri.calculando_metricas(predicoes_teste, y_teste, y_teste_recorrente)
metrica_treino = metri.calculando_metricas(predicoes_treino, y_treino, y_treino_recorrente)

print(metrica_teste)
print(metrica_treino)

#metrica_teste.to_csv("/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/metricas_teste.csv")
#metrica_treino.to_csv("/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/metricas_treino.csv")

index_treino = dados[dados.index <= data_divisao_treino_teste][6:].index
index_teste = dados[dados.index > data_divisao_treino_teste][4:].index

#############################################################################################################PLOTANDO METRICAS################################################################################

################################################################################################AVALIANDO MODELOS################################################################################
modelos_validos_teste = metrica_teste[metrica_teste['Variance'] > 0.05]

if modelos_validos_teste.empty:
    # Calcular a diferença absoluta entre métricas de treino e teste
    metrica_treino = metrica_treino.reindex(metrica_teste.index)
    metricas_diferenca = pd.DataFrame(index=metrica_treino.index)
    metricas_diferenca['MAE_diff'] =abs(metrica_treino['MAE'] - metrica_teste['MAE'])
    metricas_diferenca['MSE_diff'] = abs(metrica_treino['MSE'] - metrica_teste['MSE'])
    metricas_diferenca['RMSE_diff'] = abs(metrica_treino['RMSE'] - metrica_teste['RMSE'])

    # Calcular a soma das diferenças
    metricas_diferenca['total_diff'] = (metricas_diferenca['MAE_diff'] +
                                        metricas_diferenca['MSE_diff'] +
                                        metricas_diferenca['RMSE_diff'])
    # Selecionar o modelo com a menor soma das diferenças
    melhor_modelo = metricas_diferenca['total_diff'].idxmin()
    menor_diferenca = metricas_diferenca['total_diff'].min()

    print(f'Melhor modelo baseado na menor diferença entre treino e teste, com valor de {menor_diferenca}:', melhor_modelo)
    print(metricas_diferenca)
else:
    melhor_modelo = modelos_validos_teste['MAE'].idxmin()
    print(f'Melhor modelo baseado na MAE mais baixo, com valor de {metrica_teste["MAE"].min()}:',melhor_modelo)
    
#Conformal
index_treino_conformal = dados[dados.index <= data_divisao_treino_teste].index
index_teste_conformal = dados[dados.index > data_divisao_treino_teste].index

with parallel_config(backend='threading', n_jobs=2):
    if melhor_modelo == 'redes_neurais':
        conformal = ConformalRegressionPlotter(KerasTrainedRegressor(modelos_carregados[melhor_modelo]), x_treino_recorrente, x_teste_recorrente, y_treino[1:], y_teste[1:])
        y_pred, y_pis, _,_ = conformal.regressao_conformal()
        conformal.plot_prediction_intervals(index_train=index_treino_conformal, index_test=index_teste_conformal,title=f'Predição Intervals {melhor_modelo}',save=True,diretorio=path_codigos_rodando+f'/avaliacao_modelos/regressao_conforma_teste_{variavel_predicao}.png')
    else:
        conformal = ConformalRegressionPlotter(modelos_carregados[melhor_modelo], x_treino, x_teste, y_treino, y_teste)
        y_pred, y_pis, _,_ = conformal.regressao_conformal()
        conformal.plot_prediction_intervals(index_train=index_treino_conformal, index_test=index_teste_conformal,title=f'Predição Intervals {melhor_modelo}',save=True,diretorio=path_codigos_rodando+f'/avaliacao_modelos/regressao_conforma_teste_{variavel_predicao}.png')

dados_corformal = {
    'data': index_teste[-1],
    'predicao': y_pred,
    'intervalo_lower': y_pis.squeeze()[0:,0],
    'intervalo_upper': y_pis.squeeze()[0:,1],}    
## Predicao futuro
with parallel_config(backend='threading', n_jobs=2):
    predicao = Predicao(x_treino,y_treino,tratando,dados,melhor_modelo,modelos_carregados[melhor_modelo],periodo='Mensal',coluna=variavel_predicao)
    
dados_predicao_futuro, _, index_futuro = predicao.criando_dados_futuros()
dados_predicao = predicao.criando_dataframe_predicoes()
#dados_predicao.to_csv('/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/dados_predicao.csv',index=False)
print(dados_predicao)
data_predicao,intervalo_lower,intervalo_upper,predicao_proximo_mes = predicao.predicao_ultimo_periodo()
dados_futuro = {
    'data': data_predicao,
    'intervalo_lower': intervalo_lower,
    'intervalo_upper': intervalo_upper,
    'predicao': predicao_proximo_mes}
print(f'Data da predição:{data_predicao}, Valor da predição:{predicao_proximo_mes}, Intervalo de predição [lower,upper] :{intervalo_lower,intervalo_upper}')
predicao.plotando_predicoes(save=True,diretorio=path_codigos_rodando+f'/avaliacao_modelos/predicao_futuro_{variavel_predicao}.png')

###salvando os dados
dados_salvos = {}
dados_salvos['dados'] = dados
dados_salvos['y_treino'] = y_treino
dados_salvos['x_treino'] = x_treino
dados_salvos['y_treino_recorrente'] = y_treino_recorrente
dados_salvos['x_treino_recorrente'] = x_treino_recorrente
dados_salvos['x_teste'] = x_teste
dados_salvos['y_teste'] = y_teste
dados_salvos['y_teste_recorrente'] = y_teste_recorrente
dados_salvos['x_teste_recorrente'] = x_teste_recorrente
dados_salvos['metrica_teste'] = metrica_teste
dados_salvos['metrica_treino'] = metrica_treino
dados_salvos['predicoes_treino'] = predicoes_treino
dados_salvos['predicoes_teste'] = predicoes_teste
dados_salvos['index_treino'] = index_treino
dados_salvos['index_teste'] = index_teste
dados_salvos['melhor_modelo'] = melhor_modelo
dados_salvos['dados_conformal'] = dados_corformal
dados_salvos['dados_predicao'] = dados_predicao
dados_salvos['dados_futuro'] = dados_futuro
dados_salvos['modelos_carregados'] = list(modelos_carregados.keys())

with open(path_codigos_rodando+f'/avaliacao_modelos/apresentacao_streamlit/dados_salvos_{variavel_predicao}.pkl', 'wb') as f:
    pickle.dump(dados_salvos, f)
    
print('Avaliacao concluida com sucesso!')
