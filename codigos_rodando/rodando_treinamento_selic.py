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
    "IPCA-DP": 16122,}

variaveis_ibge = {
    'ipca': {'codigo': 1737, 'territorial_level': '1', 'ibge_territorial_code': 'all', 'variable': '63'},}

variavel_predicao = 'selic'

dados_bcb = True
dados_ibge = True
dados_expectativas_inflacao = True
dados_metas_inflacao = True
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
                                     dados_metas_inflacao=dados_metas_inflacao, 
                                     dados_ibge_link=dados_ibge_link)

tratando_scaler = True
tratando_pca = True
tratando_dummy_covid = True
tratando_defasagens = True
tratando_datas = True
tratando_estacionaridade = True

tratando = TratandoDados(dados,
                         n_components=10,
                         numero_defasagens=4,)

x_treino, x_teste, y_treino, y_teste,pca, scaler = tratando.tratando_dados(scaler=tratando_scaler, 
                                                                           pca=tratando_pca, 
                                                                           covid=tratando_dummy_covid, 
                                                                           datas=tratando_datas, 
                                                                           defasagens=tratando_defasagens, 
                                                                           estacionaridade=tratando_estacionaridade)

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
dados_salvos['pca'] = pca
dados_salvos['scaler'] = scaler
dados_salvos['data_divisao_treino_teste'] = data_divisao_treino_teste
dados_salvos['tratando'] = tratando

with open(path_codigos_rodando+f'/avaliacao_modelos/apresentacao_streamlit/dados_treinamento_{variavel_predicao}.pkl', 'wb') as f:
    pickle.dump(dados_salvos, f)