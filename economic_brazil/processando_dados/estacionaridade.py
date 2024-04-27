import sys

sys.path.append("..")
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from pmdarima import arima
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class Estacionaridade:
    def test_kpss_adf(self, dados):
        # https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
        def kpss_test(timeseries):
            # print("Results of KPSS Test:")
            kpsstest = kpss(timeseries, regression="c", nlags="auto")
            kpss_output = pd.Series(
                kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
            )
            for key, value in kpsstest[3].items():
                kpss_output["Critical Value (%s)" % key] = value
            return kpss_output

        def adf_test(timeseries):
            # print("Results of Dickey-Fuller Test:")
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

        test_est = pd.DataFrame(
            index=dados.columns,
            columns=[
                "Teste_kpss",
                "Estacionaria_ou_nâo_kpss",
                "Teste_adf",
                "Estacionaria_ou_nâo_adf",
            ],
        )
        for k in dados.columns:
            tes_kpss = kpss_test(dados[k].dropna())
            test_est.loc[k, test_est.columns[0]] = tes_kpss[1]
            tes_adf = adf_test(dados[k].dropna())
            test_est.loc[k, test_est.columns[2]] = tes_adf[1]
        test_est["Estacionaria_ou_nâo_kpss"] = np.where(
            test_est["Teste_kpss"] >= 0.05, "Estacionaria", "Nâo estacionaria"
        )
        test_est["Estacionaria_ou_nâo_adf"] = np.where(
            test_est["Teste_adf"] <= 0.05, "Estacionaria", "Nâo estacionaria"
        )
        return test_est

    def report_ndiffs(self, dados, test=None, alpha=0.05):
        if test is None:
            test = ["kpss", "adf", "pp"]
        dat_ndifis = pd.DataFrame(index=dados.columns)
        for i in test:
            dat_n = []
            for j in dados.columns:
                try:
                    dat_n.append(arima.ndiffs(dados[j].dropna(), alpha, test=i))
                except ValueError:
                    dat_n.append(0)
            dat_ndifis[i] = dat_n
        result = []
        for k in range(len(dat_ndifis)):
            result.append(
                np.where(
                    dat_ndifis.iloc[k, 0] == dat_ndifis.iloc[k, 1],
                    dat_ndifis.iloc[k, 0],
                    np.where(
                        dat_ndifis.iloc[k, 1] == dat_ndifis.iloc[k, 2],
                        dat_ndifis.iloc[k, 1],
                        np.where(
                            dat_ndifis.iloc[k, 0] == dat_ndifis.iloc[k, 2],
                            dat_ndifis.iloc[k, 2],
                            np.where(
                                dat_ndifis.iloc[k, 0]
                                != dat_ndifis.iloc[k, 1]
                                != dat_ndifis.iloc[k, 2],
                                dat_ndifis.iloc[k, 0],
                                "",
                            ),
                        ),
                    ),
                )
            )
        dat_ndifis["Ndifis"] = result
        dat_ndifis.sort_values(by="Ndifis", ascending=False, inplace=True)
        dat_ndifis["Ndifis"] = dat_ndifis["Ndifis"].astype(int)
        return dat_ndifis

    def plot_test_stationarity(self, timeseries):
        for i in range(len(timeseries.columns)):
            # Determing rolling statistics
            rolmean = (
                pd.Series(timeseries.iloc[:, i]).rolling(window=12).mean().dropna()
            )
            rolstd = pd.Series(timeseries.iloc[:, i]).rolling(window=12).std().dropna()
            # Plot rolling statistics:
            # pylint: disable=unused-variable
            orig = plt.plot(
                timeseries.iloc[:, i], color="blue", label="Original"
            )  
            # pylint: disable=unused-variable
            mean = plt.plot(
                rolmean, color="red", label="Rolling Mean"
            ) 
            std = plt.plot(
                rolstd, color="black", label="Rolling Std"
            )  
            # pylint: disable=unused-variable
            plt.legend(loc="best")
            plt.title(
                f"Rolling Mean & Standard Deviation na variavel {timeseries.columns[i]}"
            )
            plt.show(block=False)

            # Perform Dickey-Fuller test:
            print(f"Results of Dickey-Fuller Test:Coluna {timeseries.columns[i]}")
            dftest = adfuller(timeseries.iloc[:, i].dropna(), autolag="AIC")
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
            print(dfoutput)

    def corrigindo_nao_estacionaridade(self, base, n_difis, valor_predicao):
        """
        Corrigindo não estacionaridade function.

        This function takes in a base dataset, a DataFrame containing the number of differences for each variable, the value of the prediction, and returns a modified dataset with non-stationary variables differenced to remove the non-stationarity.

        Parameters:
        - base (DataFrame): The base dataset.
        - n_difis (DataFrame): A DataFrame containing the number of differences for each variable.
        - valor_predicao (str): The value of the prediction.

        Returns:
        - dados_est (DataFrame): The modified dataset with non-stationary variables differenced.
        """
        dados_est = base.copy()
        for i in n_difis[n_difis["Ndifis"] >= 1].index:
            if i == valor_predicao:
                dados_est[i] = dados_est[i]
            else:
                j = 0
                while j < n_difis["Ndifis"][i]:
                    dados_est[i] = dados_est[i].diff(periods=1)
                    j = j + 1
        dados_est = dados_est.iloc[1:]
        return dados_est
