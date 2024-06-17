import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import plotly.graph_objects as go
from economic_brazil.treinamento.redes_neurais_recorrentes import RnnModel
from economic_brazil.analisando_modelos.regressao_conformal import ConformalRegressionPlotter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class KerasTrainedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y):
        # Este método não será chamado, pois o modelo já está treinado
        pass
    
    def predict(self, X):
        return self.model.predict(X).squeeze()


class Predicao:
    def __init__(self, x_treino, y_treino, tratando_dados,dados, modelo, modelo_carregado,coluna=None):
        self.x_treino = x_treino
        self.y_treino = y_treino
        self.tratando_dados = tratando_dados
        self.dados = dados
        self.modelo = modelo
        self.modelo_carregado = modelo_carregado
        self.coluna = coluna
        self.neurais = RnnModel()
        self.mascara_sklearn = KerasTrainedRegressor(self.modelo_carregado)
        self.x_treino_recorrente, self.y_treino_recorrente = self.neurais.create_dataset(self.x_treino, self.y_treino)
        self.dados_predicao_futuro, self.dados_futuro, self.index_futuro = self.tratando_dados_futuros()
        self.y_pred_best_model, self.y_pis_best_model, _, _ = self.conformal_predicoes()

    def criando_dados_futuros(self):
        dados_futuro = self.dados.iloc[-20:].copy()
        new_index = self.dados.index[-1] + pd.DateOffset(months=1)
        dados_futuro.loc[new_index] = np.nan
        dados_futuro = dados_futuro.ffill()
        index_futuro = dados_futuro.index
        dados_predicao_futuro = self.tratando_dados.dados_futuros(dados_futuro)
        return dados_predicao_futuro, dados_futuro[self.coluna].values, index_futuro

    def tratando_dados_futuros(self):
        if self.modelo == 'redes_neurais':
            dados_predicao_futuro, dados_futuro, index_futuro = self.criando_dados_futuros()
            dados_recorrente, _ = self.neurais.create_dataset(dados_predicao_futuro, dados_futuro)
            return dados_recorrente, dados_futuro, index_futuro
        else:
            dados_predicao_futuro, dados_futuro, index_futuro = self.criando_dados_futuros()
            return dados_predicao_futuro, dados_futuro, index_futuro

    def conformal_predicoes(self):
        if self.modelo == 'redes_neurais':
            conformal = ConformalRegressionPlotter(self.mascara_sklearn, self.x_treino_recorrente, self.dados_predicao_futuro[-9:], self.y_treino_recorrente, self.dados_futuro[-9:])
        else:
            conformal = ConformalRegressionPlotter(self.modelo_carregado, self.x_treino, self.dados_predicao_futuro[-8:], self.y_treino, self.dados_futuro[-8:])
        
        y_pred_best_model, y_pis_best_model, _, _ = conformal.regressao_conformal()
        return y_pred_best_model, y_pis_best_model, _, _

    def criando_dataframe_predicoes(self):
        
        y_pis_best_model_squeezed = self.y_pis_best_model.squeeze()
        y_pred_best_model_squeezed = self.y_pred_best_model.squeeze()
        if self.modelo == 'redes_neurais':
            index_futuro_adjusted = self.index_futuro[-9:]
        else:
            index_futuro_adjusted = self.index_futuro[-8:]
        resultados_best = pd.DataFrame({
            'intervalo_lower': np.round(y_pis_best_model_squeezed[:, 0],2),
            'intervalo_upper': np.round(y_pis_best_model_squeezed[:, 1],2),
            'predicao': np.round(y_pred_best_model_squeezed,2)
        }, 
            index=index_futuro_adjusted)
        return resultados_best
    
    def predicao_ultimo_periodo(self):
        dados_predicao = self.criando_dataframe_predicoes()
        data = dados_predicao.index[-1].strftime('%Y-%m-%d')
        intervalo_lower = dados_predicao['intervalo_lower'][-1]
        intervalo_upper = dados_predicao['intervalo_upper'][-1]
        predicao_proximo_mes = dados_predicao['predicao'][-1]
        return data,intervalo_lower,intervalo_upper,predicao_proximo_mes

    def plotando_predicoes(self, save=False, diretorio=None):
        resultados_best = self.criando_dataframe_predicoes()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=resultados_best.index, y=resultados_best['predicao'], mode='lines', name='Predições'))
        fig.add_trace(
            go.Scatter(
                x=resultados_best.index,
                y=resultados_best['intervalo_lower'],
                fill=None,
                mode="lines",
                line=dict(color="lightgrey"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=resultados_best.index,
                y=resultados_best['intervalo_upper'],
                fill="tonexty",
                mode="lines",
                line=dict(color="lightgrey"),
                showlegend=False,
            )
        )
        data_legenda = resultados_best.index[-1].strftime('%Y-%m-%d')
        fig.update_layout(title=f'Predições de {data_legenda}', xaxis_title='Anos', yaxis_title='Valores')
        if save:
            fig.write_html(diretorio)
            plt.close()
        else:
            fig.show()
