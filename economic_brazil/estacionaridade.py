import sys
sys.path.append('..')
from economic_brazil.economic_data_brazil import data_economic
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")


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

class Estacionaridade():
    def test_stationarity(self,dados,plot=True):
        def kpss_test(dados):
            kpss_output = pd.Series(
                stats.kpss(dados, regression="c").statistic,
                index=["Test Statistic"],
            )
            kpss_output["p-value"] = stats.kpss(dados, regression="c").pvalue
            kpss_output["Lags Used"] = stats.kpss(dados, regression="c").nlag
            kpss_output["Critical Values"] = pd.Series(
                stats.kpss(dados, regression="c").critical_values
            )
            return kpss_output

        def adf_test(timeseries):
            adf_output = pd.Series(
                stats.adfuller(dados, autolag="AIC").statistic,
                index=["Test Statistic"],
            )
            adf_output["p-value"] = stats.adfuller(dados, autolag="AIC").pvalue
            adf_output["Num Lags Used"] = stats.adfuller(dados, autolag="AIC").nobs
            adf_output["Num Observations Used"] = stats.adfuller(
                timeseries, autolag="AIC"
            ).nobs
            adf_output["Critical Values"] = pd.Series(
                stats.adfuller(dados, autolag="AIC").critical_values
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
    
    def report_ndiffs (self,dados,test=['kpss','adf', 'pp'],alpha=0.05):
        dat_ndifis = pd.DataFrame(index=dados.columns)
        for i in test:
            dat_n = []
            for j in dados.columns:
                try:
                    dat_n.append(arima.ndiffs(dados[j].dropna(),alpha,test=i))
                except:
                    dat_n.append(0)
            dat_ndifis[i] = dat_n
        result = []
        for k in range(len(dat_ndifis)):
            result.append(np.where(dat_ndifis.iloc[k,0]==dat_ndifis.iloc[k,1],dat_ndifis.iloc[k,0],
                np.where(dat_ndifis.iloc[k,1]==dat_ndifis.iloc[k,2],dat_ndifis.iloc[k,1],
                        np.where(dat_ndifis.iloc[k,0]==dat_ndifis.iloc[k,2],dat_ndifis.iloc[k,2],
                        np.where(dat_ndifis.iloc[k,0]!=dat_ndifis.iloc[k,1]!=dat_ndifis.iloc[k,2],dat_ndifis.iloc[k,0],'')))))
        dat_ndifis['Ndifis'] = result
        dat_ndifis.sort_values(by='Ndifis',ascending=False,inplace=True)
        dat_ndifis['Ndifis'] = dat_ndifis['Ndifis'].astype(int)
        return dat_ndifis