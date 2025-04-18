import sys
sys.path.append('..')
from economic_brazil.coleta_dados.economic_data_brazil import EconomicBrazil
from economic_brazil.processando_dados.tratando_dados import TratandoDados
from economic_brazil.treinamento.treinamento_algoritimos import TreinandoModelos
import warnings
import pickle
import os

warnings.filterwarnings("ignore", category=UserWarning)

path_codigos_rodando = os.path.join(os.getcwd())

SELIC_CODES = {
    "selic": 4189,
    "IPCA-EX2": 27838,
    "IPCA-EX3": 27839,
    "IPCA-MS": 4466,
    "IPCA-MA": 11426,
    "IPCA-EX0": 11427,
    "IPCA-EX1": 16121,
    "IPCA-DP": 16122,
    "divida_liquida_spc": 4513,
    'metas_inflacao': 13521,
    'indice_expectativas_futuras': 4395,
    "divida_liquida_spc": 4513,
    'metas_inflacao': 13521,
    'indice_expectativas_futuras': 4395,
    'indice_cambio_real_efetiva':11752,
    'rendimento_medio_real_trabalhadores':24381,
    'massa_rendimento_real_trabalhadores':28544}

variaveis_ibge = {
    'ipca': {'codigo': 1737, 'territorial_level': '1', 'ibge_territorial_code': 'all', 'variable': '63'},}

variavel_predicao = 'selic'

dados_bcb = True
dados_ibge = True
dados_expectativas_inflacao = True
dados_ibge_link = True
dados_ipeadata = False
dados_google_trends = False
dados_fred =False

economic_brazil = EconomicBrazil(codigos_banco_central=SELIC_CODES, codigos_ibge=variaveis_ibge, data_inicio="2000-01-01")

dados = economic_brazil.dados_brazil(dados_ipeadata=dados_ipeadata, 
                                     dados_bcb= dados_bcb,
                                     dados_google_trends=dados_google_trends, 
                                     dados_expectativas_inflacao=dados_expectativas_inflacao, 
                                     dados_ibge_codigos=dados_ibge, 
                                     dados_ibge_link=dados_ibge_link)

tratando_scaler = True
tratando_pca = False
tratando_dummy_covid = True
tratando_defasagens = True
tratando_datas = True
tratando_estacionaridade = True
tratando_rfe = True
tratando_smart_correlation = True
tratando_variance = True
tratando_estacionaridade_log = True
data_divisao = '2020-06-01'
tratando = TratandoDados(dados,
                         coluna_label=variavel_predicao,
                         data_divisao=data_divisao,
                         numero_defasagens=4,
                         n_features_to_select=100,
                         scaler=tratando_scaler, 
                        pca=tratando_pca, 
                        covid=tratando_dummy_covid, 
                        datas=tratando_datas, 
                        defasagens=tratando_defasagens, 
                        estacionaridade=tratando_estacionaridade,
                        rfe=tratando_rfe,
                        smart_correlation=tratando_smart_correlation,
                        variancia=tratando_variance,
                        estacionaridade_log=tratando_estacionaridade_log)

x_treino, x_teste, y_treino, y_teste, pca , scaler , rfe_model, variance_model, smart_model = tratando.tratando_dados()

data_divisao_treino_teste = tratando.data_divisao_treino_teste()

tuning = TreinandoModelos(x_treino, y_treino, x_teste, y_teste,diretorio=f'../codigos_rodando/modelos_salvos/modelos_{variavel_predicao}/',salvar_modelo=True)

modelo_redes_neurais = True
modelo_cast = True
modelo_sarimax = False
modelo_gradient_boosting = True
modelo_regresao_linear = True
modelo_xgboost = True

modelos_tunning = tuning.treinar_modelos(redes_neurais=modelo_redes_neurais,
                                         cat_boost=modelo_cast,
                                         sarimax=modelo_sarimax,
                                         gradiente_boosting=modelo_gradient_boosting,
                                         regressao_linear=modelo_regresao_linear,
                                         xg_boost=modelo_xgboost
                                         )

dados_salvos = {}
dados_salvos['dados'] = dados
dados_salvos['x_treino'] = x_treino
dados_salvos['x_teste'] = x_teste
dados_salvos['y_treino'] = y_treino
dados_salvos['y_teste'] = y_teste
if pca:
    dados_salvos['pca'] = pca
if scaler:
    dados_salvos['scaler'] = scaler
if rfe_model:
    dados_salvos['rfe_model'] = rfe_model
if variance_model:
    dados_salvos['variance_model'] = variance_model
if smart_model:
    dados_salvos['smart_model'] = smart_model
dados_salvos['data_divisao_treino_teste'] = data_divisao_treino_teste
dados_salvos['tratando'] = tratando

with open(path_codigos_rodando+f'/avaliacao_modelos/apresentacao_streamlit/dados_treinamento_{variavel_predicao}.pkl', 'wb') as f:
    pickle.dump(dados_salvos, f)