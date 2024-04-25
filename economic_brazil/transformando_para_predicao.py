import sys
sys.path.append('..')
from economic_brazil.economic_data_brazil import data_economic
import pandas as pd
import numpy as np
from codigos_graficos import Graficos
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

data = data_economic(salvar=True, formato="csv",diretorio="economic_data_brazil.csv")


import pandas as pd
import numpy as np

def criando_dummy_covid(dados, inicio_periodo: str, fim_periodo: str) -> pd.DataFrame:
    """
    Cria uma coluna 'dummy_covid' no DataFrame 'dados' com valores 1 para os registros dentro do período de datas especificado e 0 para os de fora.

    Args:
        dados (pd.DataFrame): O DataFrame contendo os dados.
        inicio_periodo (str): A data de início do período em formato 'YYYY-MM-DD'.
        fim_periodo (str): A data de fim do período em formato 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: O DataFrame 'dados' com a coluna 'dummy_covid' adicionada.
    """
    periodo = pd.date_range(start=inicio_periodo, end=fim_periodo)
    dados['dummy_covid'] = np.where(dados.index.isin(periodo), 1, 0)
    return dados


def backcasting_nan(dados):
    """
    Fills missing values in the input DataFrame with interpolated values using the XGBoost regressor model.
    
    Parameters:
        dados (pd.DataFrame): The input DataFrame containing the data to be processed.
        
    Returns:
        pd.DataFrame: The processed DataFrame with missing values filled in.
    """
    colunas = dados.columns.tolist()
    np.random.seed(0)
    dados_bac = dados.copy()
    
    for i in colunas:
        dados_interpolacao = dados.loc[:dados[i].last_valid_index()]

        treino_interpolacao = dados_interpolacao.dropna(subset=[i])
        predicao_interpolacao = dados_interpolacao[dados_interpolacao[i].isnull()]

        Y_treino = treino_interpolacao[i]
        treino_interpolacao = treino_interpolacao.drop(i, axis=1)
        predicao_interpolacao = predicao_interpolacao.drop(i, axis=1)

        predicao_interpolacao = predicao_interpolacao.dropna(thresh=len(predicao_interpolacao) - 4, axis=1)
        Y_treino = Y_treino[Y_treino.index.isin(treino_interpolacao.index)]
        treino_interpolacao = treino_interpolacao.loc[:, predicao_interpolacao.columns]

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', XGBRegressor(random_state=0))
        ])
        
        try:
            pipeline.fit(treino_interpolacao, Y_treino)
            predicao = pipeline.predict(predicao_interpolacao)
            interpolados = pd.DataFrame(predicao, index=predicao_interpolacao.index, columns=[i])
            dados_bac.update(interpolados)
        except Exception as e:
            print(f"Error processing column {i}: {str(e)}")
    
    return dados_bac


##write function for stacionarity test

def test_kpss_adf(dados):
    #https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    def kpss_test(timeseries):
        
        kpsstest = kpss(timeseries, regression="c", nlags="auto")
        kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
        )
        for key, value in kpsstest[3].items():
            kpss_output["Critical Value (%s)" % key] = value
        return kpss_output

    def adf_test(timeseries):
        
        #print("Results of Dickey-Fuller Test:")
        dftest = adfuller(timeseries, autolag="AIC")
        dfoutput = pd.Series(
        dftest[0:4],
        index=[
                "Test Statistic",
                "p-value",
                "#Lags Used",
                "Number of Observations Used",
            ],
        )
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)" % key] = value
        return dfoutput


    test_est = pd.DataFrame(index=dados.columns,columns = ['Teste_kpss', 'Estacionaria_ou_nâo_kpss','Teste_adf','Estacionaria_ou_nâo_adf'])
    for k in dados.columns:
        tes_kpss = kpss_test(dados[k].dropna())
        test_est.loc[k,test_est.columns[0]] = tes_kpss[1]
        tes_adf = adf_test(dados[k].dropna())
        test_est.loc[k,test_est.columns[2]] = tes_adf[1]
    test_est['Estacionaria_ou_nâo_kpss']=np.where(test_est['Teste_kpss'] >= 0.05,'Estacionaria','Nâo estacionaria')
    test_est['Estacionaria_ou_nâo_adf']=np.where(test_est['Teste_adf'] <= 0.05,'Estacionaria','Nâo estacionaria')
    return test_est

import scipy.stats as stats
import pandas as pd
import numpy as np

def test_kpss_adf(dados):
    def kpss_test(timeseries):
        kpss_output = pd.Series(
            stats.kpss(timeseries, regression="c").statistic,
            index=["Test Statistic"],
        )
        kpss_output["p-value"] = stats.kpss(timeseries, regression="c").pvalue
        kpss_output["Lags Used"] = stats.kpss(timeseries, regression="c").nlag
        kpss_output["Critical Values"] = pd.Series(
            stats.kpss(timeseries, regression="c").critical_values
        )
        return kpss_output

    def adf_test(timeseries):
        adf_output = pd.Series(
            stats.adfuller(timeseries, autolag="AIC").statistic,
            index=["Test Statistic"],
        )
        adf_output["p-value"] = stats.adfuller(timeseries, autolag="AIC").pvalue
        adf_output["Num Lags Used"] = stats.adfuller(timeseries, autolag="AIC").nobs
        adf_output["Num Observations Used"] = stats.adfuller(
            timeseries, autolag="AIC"
        ).nobs
        adf_output["Critical Values"] = pd.Series(
            stats.adfuller(timeseries, autolag="AIC").critical_values
        )
        return adf_output

    test_est = pd.DataFrame(
        index=dados.columns,
        columns=["KPSS Test", "Stationary (KPSS)", "ADF Test", "Stationary (ADF)"],
    )
    test_est["KPSS Test"] = dados.apply(kpss_test)["p-value"]
    test_est["Stationary (KPSS)"] = np.where(
        test_est["KPSS Test"] >= 0.05, "Stationary", "Not Stationary"
    )
    test_est["ADF Test"] = dados.apply(adf_test)["p-value"]
    test_est["Stationary (ADF)"] = np.where(
        test_est["ADF Test"] <= 0.05, "Stationary", "Not Stationary"
    )
    return test_est
  
    



    
        
        
