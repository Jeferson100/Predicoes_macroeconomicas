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

codigos_banco_central = {
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
    'metas_inflacao': 13521,
    'indice_expectativas_futuras': 4395
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

############################################################################## VARIAVEL PREDICAO ################################################################
variavel_predicao = 'ipca'

dados_bcb = True
dados_ibge = True
dados_expectativas_inflacao = True
dados_ibge_link = True
dados_ipeadata = True
dados_google_trends = True
dados_fred =False



economic_brazil = EconomicBrazil(codigos_banco_central=codigos_banco_central, 
                                 codigos_ibge=variaveis_ibge, 
                                 codigos_ipeadata=codigos_ipeadata_padrao, 
                                 lista_termos_google_trends=lista, 
                                 data_inicio="2000-01-01")

dados = economic_brazil.dados_brazil(dados_bcb= dados_bcb,
                                     dados_expectativas_inflacao=dados_expectativas_inflacao, 
                                     dados_ibge_codigos=dados_ibge, 
                                     dados_ibge_link=dados_ibge_link, 
                                     dados_ipeadata=dados_ipeadata, 
                                     dados_google_trends=dados_google_trends,)


print(dados.shape)

tratando_scaler = True
tratando_pca = False
tratando_dummy_covid = True
tratando_defasagens = True
tratando_datas = True
tratando_estacionaridade = True
tratando_rfe = True
tratando_smart_correlation = True
tratando_variance = True
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
                        variancia=tratando_variance)

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