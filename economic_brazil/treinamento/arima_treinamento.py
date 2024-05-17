import pandas as pd
import statsmodels.tsa.api as tsa
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from numpy.linalg import LinAlgError
from sklearn.metrics import mean_squared_error


class Arima:
    def treinar_arima(self, dados, p=1, n=0, q=1):
        model = tsa.ARIMA(dados, order=(p, n, q))
        model_fit = model.fit()
        print(model_fit.summary())
        return model_fit

    def encontrar_parametros_arima(self, dados, divisao=120):
        train_size = divisao
        results = {}
        y_true = dados[train_size:]
        for p in range(5):
            for q in range(5):
                aic, bic = [], []
                if p == 0 and q == 0:
                    continue
                # print(p, q)
                convergence_error = stationarity_error = 0
                y_pred = []
                for T in range(train_size, len(dados)):
                    #train_set = dados.iloc[T - train_size : T]
                    train_set = dados[T - train_size : T]
                    try:
                        model = tsa.ARIMA(endog=train_set, order=(p, 0, q)).fit()
                    except LinAlgError:
                        convergence_error += 1
                    except ValueError:
                        stationarity_error += 1

                    # forecast, _, _ = model.forecast(steps=1)
                    forecast = model.forecast(steps=1)
                    y_pred.append(forecast[0])
                    aic.append(model.aic)
                    bic.append(model.bic)

                result = (
                    pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
                    .replace(np.inf, np.nan)
                    .dropna()
                )

                rmse = np.sqrt(
                    mean_squared_error(y_true=result.y_true, y_pred=result.y_pred)
                )

                results[(p, q)] = [
                    rmse,
                    np.mean(aic),
                    np.mean(bic),
                    convergence_error,
                    stationarity_error,
                ]
                arma_results = pd.DataFrame(results).T
                arma_results.columns = [
                    "RMSE",
                    "AIC",
                    "BIC",
                    "convergence",
                    "stationarity",
                ]
                arma_results.index.names = ["p", "q"]
                rank_arima = (
                    arma_results.rank().loc[:, ["RMSE", "BIC"]].mean(1).nsmallest(5)
                )
                valor_p = rank_arima.index[0][0]
                valor_q = rank_arima.index[0][1]

        return rank_arima, valor_p, valor_q

    def prever_arima(self, modelo, steps=1):
        predicao = modelo.forecast(steps)
        return predicao
    
class Sarimax:
    def treinar_sarimax(self, y_treino, x_treino, p=1, d=0,q=1):
        model = tsa.SARIMAX(endog=y_treino, exog=x_treino, order=(p, d, q))
        model_fit = model.fit()
        print(model_fit.summary())
        return model_fit
    
    def encontrar_parametros_sarimax(self, x, y, divisao=120):
        train_size = divisao
        results = {}
        y_true = y[train_size:]
        for p in range(5):
            for q in range(5):
                aic, bic = [], []
                if p == 0 and q == 0:
                    continue
                convergence_error = stationarity_error = 0
                y_pred = []
                for T in range(train_size, len(x)):
                    #train_set = dados.iloc[T - train_size : T]
                    train_set = x[T - train_size : T]
                    y_true_set = y[T - train_size : T]       
                            
                    model = SARIMAX(endog=y_true_set,exog=train_set, order=(p, 0, q)).fit(disp=False)

                    # forecast, _, _ = model.forecast(steps=1)
                    forecast = model.predict(start=0,end=y_true.shape[0],exog=train_set)
                    y_pred.append(forecast[0])
                    aic.append(model.aic)   
                    bic.append(model.bic)

                result = (
                            pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
                            .replace(np.inf, np.nan)
                            .dropna()
                        )

                rmse = np.sqrt(
                            mean_squared_error(y_true=result.y_true, y_pred=result.y_pred)
                        )

                results[(p, q)] = [
                            rmse,
                            np.mean(aic),
                            np.mean(bic),
                            convergence_error,
                            stationarity_error,
                        ]
                arma_results = pd.DataFrame(results).T
                arma_results.columns = [
                            "RMSE",
                            "AIC",
                            "BIC",
                            "convergence",
                            "stationarity",
                        ]
                arma_results.index.names = ["p", "q"]
                rank_arima = (
                            arma_results.rank().loc[:, ["RMSE", "BIC"]].mean(1).nsmallest(5)
                        )
                valor_p = rank_arima.index[0][0]
                valor_q = rank_arima.index[0][1]
                
        return rank_arima, valor_p, valor_q
    
    def prever_sarimax(self, start, end, modelo,exog):
        predicao = modelo.predict(start=start,end=end,exog=exog)
        return predicao