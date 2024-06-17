import sys
sys.path.append('..')
from economic_brazil.coleta_dados.economic_data_brazil import EconomicBrazil
from economic_brazil.processando_dados.tratando_dados import TratandoDados
from economic_brazil.treinamento.treinamento_algoritimos import TreinandoModelos
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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

economic_brazil = EconomicBrazil(codigos_banco_central=SELIC_CODES, codigos_ibge=variaveis_ibge, codigos_ipeadata=codigos_ipeadata_padrao, lista_termos_google_trends=lista, data_inicio="2000-01-01")

dados = economic_brazil.dados_brazil(dados_bcb= True, dados_expectativas_inflacao=True, dados_ibge_codigos=True, dados_metas_inflacao=True, dados_ibge_link=True, dados_ipeadata=True, dados_google_trends=True)

tratando = TratandoDados(dados,coluna_label="ipca")

x_treino, x_teste, y_treino, y_teste,pca, scaler = tratando.tratando_dados()

tuning = TreinandoModelos(x_treino, y_treino, x_teste, y_teste,diretorio='../codigos_rodando/modelos_salvos/modelos_ipca/',salvar_modelo=True)

modelos_tunning = tuning.treinar_modelos(redes_neurais=True,sarimax=False)