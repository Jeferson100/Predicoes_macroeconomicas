import sys
sys.path.append('..')
#from economic_brazil.coleta_dados.economic_data_brazil import EconomicBrazil
#from economic_brazil.processando_dados.tratando_dados import TratandoDados
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

variavel_predicao = 'ipca'

SELIC_CODES = {
    "cambio": 3698,
    "pib_mensal": 4380,
    "igp_m": 189,
    "igp_di": 190,
    "m1": 27788,
    "m2": 27810,
    "m3": 27813,
    "m4": 27815,
    "estoque_caged": 28763,
    "saldo_bc": 22707,
    "vendas_auto": 7384,
    "divida_liquida_spc": 4513,
}

variaveis_ibge = {
    "ipca": {
        "codigo": 1737,
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "63",
    },
    "custo_m2": {
        "codigo": 2296,
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "1198",
    },
    "pesquisa_industrial_mensal": {
        "codigo": 8159,
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "11599",
    },
    "pmc_volume": {
        "codigo": 8186,
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "11709",
    },
}

codigos_ipeadata_padrao = {
    "taja_juros_ltn": "ANBIMA12_TJTLN1212",
    "imposto_renda": "SRF12_IR12",
    "ibovespa": "ANBIMA12_IBVSP12",
    "consumo_energia": "ELETRO12_CEET12",
    "brent_fob": "EIA366_PBRENT366",
}

lista = [
    'seguro desemprego',

]

#economic_brazil = EconomicBrazil(codigos_banco_central=SELIC_CODES, codigos_ibge=variaveis_ibge, codigos_ipeadata=codigos_ipeadata_padrao, lista_termos_google_trends=lista, data_inicio="2000-01-01")

#dados = economic_brazil.dados_brazil(dados_bcb= True, dados_expectativas_inflacao=True, dados_ibge_codigos=True, dados_metas_inflacao=True, dados_ibge_link=True, dados_ipeadata=True, dados_google_trends=True)

arquivo_selic = path_codigos_rodando+f'/avaliacao_modelos/apresentacao_streamlit/dados_treinamento_{variavel_predicao}.pkl'
dados_carregados = pickle.load(open(arquivo_selic, 'rb'))

#tratando = TratandoDados(dados,coluna_label="ipca")
dados = dados_carregados['dados']
x_treino = dados_carregados['x_treino']
x_teste = dados_carregados['x_teste']
y_treino = dados_carregados['y_treino']
y_teste = dados_carregados['y_teste']
pca = dados_carregados['pca']
if 'pca' in dados_carregados.keys():    
    scaler = dados_carregados['scaler']
data_divisao_treino_teste = dados_carregados['data_divisao_treino_teste']
tratando = dados_carregados['tratando']



modelos_carregados = carregar(diretorio=path_codigos_rodando+f'/modelos_salvos/modelos_{variavel_predicao}/',gradiente_boosting=True, xg_boost=True, cat_boost=True, regressao_linear=True, redes_neurais=True, sarimax=False)

#Prevendo os dados de treino e teste
predi = PredicaosModelos(modelos_carregados, x_treino, y_treino, x_teste, y_teste)
predicoes_treino, predicoes_teste = predi.predicoes()
x_treino, y_treino, x_teste, y_teste, x_treino_recorrente, y_treino_recorrente, x_teste_recorrente, y_teste_recorrente = predi.return_dados()
#Calculando metricas
metri = MetricasModelosDicionario()
metrica_teste = metri.calculando_metricas(predicoes_teste, y_teste, y_teste_recorrente)
metrica_treino = metri.calculando_metricas(predicoes_treino, y_treino, y_treino_recorrente)

print(metrica_teste)
print(metrica_treino)
index_treino = dados[dados.index <= data_divisao_treino_teste][6:].index
index_teste = dados[dados.index > data_divisao_treino_teste][4:].index
# Plotando predicoes
metri.plotando_predicoes(
    y_treino,
    predicoes_treino,
    index=index_treino,
    title="Predições nos dados de treino",
    save=True,
    diretorio=path_codigos_rodando+f'/avaliacao_modelos/predicao_treino_{variavel_predicao}.png',
)

metri.plotando_predicoes(
    y_teste,
    predicoes_teste,
    index=index_teste,
    title="Predições nos dados de teste",
    save=True,
    diretorio=path_codigos_rodando+f'/avaliacao_modelos/predicao_teste_{variavel_predicao}.png',
)

metri.plotando_predicoes_go_treino_teste(
    y_treino,
    y_teste,
    predicoes_treino,
    predicoes_teste,
    index_treino,
    index_teste,
    save=True,
    diretorio=path_codigos_rodando+f'/avaliacao_modelos/predicao_treino_teste_{variavel_predicao}.png',
    type_arquivo='png'
)

##melhor modelo baseado em MENOR erro absoluto (MAE)
modelos_validos_teste = metrica_teste[metrica_teste['Variance'] > 0.1]
if modelos_validos_teste.empty:
    melhor_modelo = metrica_teste['Variance'].idxmax()
    print(f'Melhor modelo baseado na Variance mais alta, com valor de {metrica_teste["Variance"].max()}:',melhor_modelo)
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
    'data': index_teste[1:],
    'predicao': y_pred,
    'intervalo_lower': y_pis.squeeze()[0:,0],
    'intervalo_upper': y_pis.squeeze()[0:,1],}  

## Predicao futuro
with parallel_config(backend='threading', n_jobs=2):
    predicao = Predicao(x_treino,y_treino,tratando,dados,melhor_modelo,modelos_carregados[melhor_modelo],coluna=variavel_predicao)
    
dados_corformal = {
    'data': index_teste[1:],
    'predicao': y_pred,
    'intervalo_lower': y_pis.squeeze()[0:,0],
    'intervalo_upper': y_pis.squeeze()[0:,1],}    

## Predicao futuro
with parallel_config(backend='threading', n_jobs=2):
    predicao = Predicao(x_treino,y_treino,tratando,dados,melhor_modelo,modelos_carregados[melhor_modelo],coluna='ipca')
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
predicao.plotando_predicoes(save=True,diretorio=path_codigos_rodando+'/avaliacao_modelos/predicao_futuro_ipca.png')


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