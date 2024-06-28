import sys
sys.path.append('..')
from economic_brazil.coleta_dados.economic_data_brazil import EconomicBrazil
from economic_brazil.processando_dados.tratando_dados import TratandoDados
from economic_brazil.treinamento.treinamento_algoritimos import TreinandoModelos
import warnings
import pandas as pd
import pickle
import os

path_codigos_rodando = os.path.join(os.getcwd())

warnings.filterwarnings("ignore", category=UserWarning)

banco_central_codes = {
    "indice_condicoes_economicas": 4394,
    "indice_valores_garantias_imoveis_residencias_financiados": 21340,
    "venda_veiculos_concessionarias": 1378,
    "indicador_movimento_comercio_prazo": 1453,
    "indice_volume_vendas_varejo": 1455,
    "imposto_sobre_produtos": 22098
}

variaveis_ibge = {
    "ipca": {
        "codigo": '1737',
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "63",
    },
    "custo_m2": {
        "codigo": '2296',
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "1198",
    },
    "pmc_volume": {
        "codigo": '8186',
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "11709",
    },
    "producao_fisica_industrial": {
        "codigo": '8159',
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "11599",},

    "producao_fisica_para_construcao_civil": {
        "codigo": '7980',
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "11599",},
    
    "producao_soja_milho": {    
        "codigo": '6588',
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "35",},
    
    "precos_construcao_civil": {
        "codigo": '2296',
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "1198",},
    
    "volume_servicos_(pms)": {
        "codigo": '8162',
        "territorial_level": "1",
        "ibge_territorial_code": "all",
        "variable": "11621",
    }
    
    
}

codigos_ipeadata_padrao = {
    "rendimento_real_medio": "PNADC12_RRTH12",
    "pessoas_forca_trabalho": "PNADC12_FT12",
    "taxa_desocupacao": "PNADC12_TDESOC12",
    "caged_novo": "CAGED12_SALDON12",
    "caged_antigo": "CAGED12_SALDO12",
    'ibovespa': 'ANBIMA12_IBVSP12',
    "taja_juros_ltn": "ANBIMA12_TJTLN1212",
    'exportacoes': 'PAN12_XTV12',
    'importacoes': 'PAN12_MTV12',
    'm_1': 'BM12_M1MN12',
    'taxa_cambio': 'PAN12_ERV12',
    'atividade_economica': 'SGS12_IBCBR12',
}

lista = [
    'seguro desemprego']

indicadores_ibge_link = {
    "pib": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6564/p/all/c11255/90707/d/v6564%201/l/v,p,t%2Bc11255&verUFs=false&verComplementos2=false&verComplementos1=false&omitirIndentacao=false&abreviarRotulos=false&exibirNotas=false&agruparNoCabecalho=false",
    "despesas_publica": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93405/d/v6561%201/l/v,p%2Bc11255,t",
    "capital_fixo": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela5932.xlsx&terr=N&rank=-&query=t/5932/n1/all/v/6561/p/all/c11255/93406/d/v6561%201/l/v,p%2Bc11255,t",
    "producao_industrial_manufatureira": "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela8158.xlsx&terr=N&rank=-&query=t/8158/n1/all/v/11599/p/all/c543/129278/d/v11599%205/l/v,p%2Bc543,t",
    'soja' :'https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela6588.xlsx&terr=N&rank=-&query=t/6588/n1/all/v/35/p/all/c48/0,39443/l/v,p%2Bc48,t',
    'milho_1' :'https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela6588.xlsx&terr=N&rank=-&query=t/6588/n1/all/v/35/p/all/c48/0,39441/l/v,p%2Bc48,t',
    'milho_2' :'https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela6588.xlsx&terr=N&rank=-&query=t/6588/n1/all/v/35/p/all/c48/0,39442/l/v,p%2Bc48,t',
    'pms': 'https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela8162.xlsx&terr=N&rank=-&query=t/8162/n1/all/v/11622/p/all/c11046/56726/c12355/107071/d/v11622%205/l/v,p%2Bc11046,t%2Bc12355'
}

codigos_fred = {'nasdaq100':'NASDAQ100',
                'taxa_cambio_efetiva':'RBBRBIS',
                'cboe_nasdaq':'VXNCLS',
                'taxa_juros_interbancaria':'IRSTCI01BRM156N',
                'atividade_economica_eua':'USPHCI',
                'indice_confianca_manufatura':'BSCICP03BRM665S',
                'indice_confianca_exportadores':'BSXRLV02BRM086S',
                'indice_tendencia_emprego':'BRABREMFT02STSAM',
                'indice_confianca_consumidor':'CSCICP03BRM665S',
                'capacidade_instalada':'BSCURT02BRM160S',}

############################################################################## VARIAVEL PREDICAO ################################################################
variavel_predicao = 'pib'


################################################################################ DADOS ############################################################################

dados_bcb = True
dados_ibge = True
dados_expectativas_inflacao = False
dados_metas_inflacao = False
dados_ibge_link = True
dados_ipeadata = True
dados_google_trends = True
dados_fred =True

economic_brasil = EconomicBrazil(codigos_banco_central=banco_central_codes, 
                       codigos_ibge=variaveis_ibge,
                       codigos_ipeadata=codigos_ipeadata_padrao,
                       lista_termos_google_trends=lista,
                       codigos_ibge_link=indicadores_ibge_link,
                       codigos_fred=codigos_fred,
                       data_inicio="1997-01-01")

dados = economic_brasil.dados_brazil(
                dados_bcb=dados_bcb,
                dados_ipeadata=dados_ipeadata,
                dados_fred=dados_fred,
                dados_metas_inflacao=dados_metas_inflacao,
                dados_google_trends=dados_google_trends,
                dados_expectativas_inflacao=dados_expectativas_inflacao,
                dados_ibge_link=dados_ibge_link,
                dados_ibge_codigos=dados_ibge,
    
)
#Colcando os dados em trimestre
dados_trimestral = dados.resample('QS').median()

############################################################################## TRATAMENTO ############################################################################


tratando_scaler = True
tratando_pca = True
tratando_dummy_covid = True
tratando_defasagens = True
tratando_datas = True
tratando_estacionaridade = True

data_divisao = '2020-06-01'

tratando = TratandoDados(dados_trimestral,
                         coluna_label=variavel_predicao,
                         data_divisao=data_divisao,
                         n_components=10,
                         numero_defasagens=4)

x_treino, x_teste, y_treino, y_teste,pca, scaler = tratando.tratando_dados(scaler=tratando_scaler, 
                                                                           pca=tratando_pca, 
                                                                           covid=tratando_dummy_covid, 
                                                                           datas=tratando_datas, 
                                                                           defasagens=tratando_defasagens, 
                                                                           estacionaridade=tratando_estacionaridade,
                                                                           )

data_divisao_treino_teste = tratando.data_divisao_treino_teste()

################################################################################## TREINAMENTO ############################################################################
param_grid_gradiente = {
                    "n_estimators": [2, 10,20,30, 50, 100, 200, 300, 500],
                    "learning_rate": [0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1, 1.25, 1.5,1.75,],
                    "max_depth": [1, 3, 5, 7, 9],
                }

param_grid_xgboost = {
                    "n_estimators": [2, 10,20,30, 50, 100, 200, 300, 500],
                    "learning_rate": [0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1, 1.25, 1.5,1.75,],
                    "max_depth": [1, 3, 5, 7, 9],
                }

param_grid_catboost = {
                    "iterations": [2, 10,20,30, 50, 100, 200, 300, 500],
                    "learning_rate": [0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1],
                    "depth": [1, 3, 5, 7, 9,10,15],
                }


tuning = TreinandoModelos(x_treino, 
                          y_treino, 
                          x_teste, 
                          y_teste,
                          tuning_grid_search=False,
                          tuning_random_search=False,
                          tuning_bayes_search=True,
                          numero_divisoes = 5,
                          diretorio=f'../codigos_rodando/modelos_salvos/modelos_{variavel_predicao}/',
                          salvar_modelo=True)

modelo_redes_neurais = True
modelo_cast = True
modelo_sarimax = False
modelo_gradient_boosting = True
modelo_regresao_linear = True
modelo_xgboost = True
redes_neurais_tuning = False

modelos_tunning = tuning.treinar_modelos(redes_neurais=modelo_redes_neurais,
                                         cat_boost=modelo_cast,
                                         sarimax=modelo_sarimax,
                                         gradiente_boosting=modelo_gradient_boosting,
                                         regressao_linear=modelo_regresao_linear,
                                         xg_boost=modelo_xgboost,
                                         redes_neurais_tuning=redes_neurais_tuning,
                                         param_grid_catboost=param_grid_catboost,
                                        param_grid_xgboost=param_grid_xgboost,
                                        param_grid_gradiente=param_grid_gradiente
                                         )

################################################################################### SALVANDO DADOS ############################################################################
dados_salvos = {}
dados_salvos['dados'] = dados_trimestral
dados_salvos['x_treino'] = x_treino
dados_salvos['x_teste'] = x_teste
dados_salvos['y_treino'] = y_treino
dados_salvos['y_teste'] = y_teste
if pca:
    dados_salvos['pca'] = pca
if scaler:
    dados_salvos['scaler'] = scaler
dados_salvos['data_divisao_treino_teste'] = data_divisao_treino_teste
dados_salvos['tratando'] = tratando

with open(path_codigos_rodando+f'/avaliacao_modelos/apresentacao_streamlit/dados_treinamento_{variavel_predicao}.pkl', 'wb') as f:
    pickle.dump(dados_salvos, f)
