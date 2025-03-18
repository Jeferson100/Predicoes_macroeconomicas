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
from typing import Any, Optional
warnings.filterwarnings("ignore")

class KerasTrainedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model: Any) -> None:
        self.model = model
        
    def fit(self, X: Any, y: Any)  -> None:
        # Este método não será chamado, pois o modelo já está treinado
        pass
    
    def predict(self, X: Any) -> Any:
        return self.model.predict(X).squeeze()


class Predicao:
    def __init__(self, x_treino: pd.DataFrame, y_treino: pd.DataFrame, tratando_dados: Any,dados: pd.DataFrame, modelo: str, modelo_carregado: Any, periodo:Optional[str]=None, coluna:Optional[str]=None):
        self.x_treino = x_treino
        self.y_treino = y_treino
        self.tratando_dados = tratando_dados
        self.dados = dados
        self.modelo = modelo
        self.modelo_carregado = modelo_carregado
        self.periodo = periodo
        self.coluna = coluna
        #self.tratando_pca = tratando_pca
        #self.tratando_scaler = tratando_scaler
        self.neurais = RnnModel()
        self.mascara_sklearn = KerasTrainedRegressor(self.modelo_carregado)
        self.x_treino_recorrente, self.y_treino_recorrente = self.neurais.create_dataset(self.x_treino, self.y_treino)
        self.dados_predicao_futuro, self.dados_futuro, self.index_futuro = self.tratando_dados_futuros()
        self.y_pred_best_model, self.y_pis_best_model, _, _ = self.conformal_predicoes()

    def criando_dados_futuros(self) -> Any:
        if self.periodo == None:
            self.periodo = 'Mensal'
            
        dados_futuro = self.dados.iloc[-20:].copy()
        
        if self.periodo == 'Mensal':
            new_index = dados_futuro.index[-1] + pd.DateOffset(months=1)
        elif self.periodo == 'Anual':
            new_index = dados_futuro.index[-1] + pd.DateOffset(years=1)
        elif self.periodo == 'Trimestral':
            new_index = dados_futuro.index[-1] + pd.DateOffset(months=3)
        elif self.periodo == 'Semestral':
            new_index = dados_futuro.index[-1] + pd.DateOffset(months=6)
        else:
            print('Periodo invalido:Apenas mensal, anual, trimestral e semestral')

        dados_futuro.loc[new_index] = np.nan #type: ignore
        dados_futuro = dados_futuro.ffill()
        index_futuro = dados_futuro.index
        dados_predicao_futuro = self.tratando_dados.dados_futuros(dados_futuro)
        return dados_predicao_futuro, dados_futuro[self.coluna].values, index_futuro

    def tratando_dados_futuros(self) -> Any:
        if self.modelo == 'redes_neurais':
            dados_predicao_futuro, dados_futuro, index_futuro = self.criando_dados_futuros()
            dados_recorrente, _ = self.neurais.create_dataset(dados_predicao_futuro, dados_futuro)
            return dados_recorrente, dados_futuro, index_futuro
        else:
            dados_predicao_futuro, dados_futuro, index_futuro = self.criando_dados_futuros()
            return dados_predicao_futuro, dados_futuro, index_futuro

    def conformal_predicoes(self) -> Any:
        if self.modelo == 'redes_neurais':
            conformal = ConformalRegressionPlotter(self.mascara_sklearn, self.x_treino_recorrente, self.dados_predicao_futuro[-9:], self.y_treino_recorrente, self.dados_futuro[-9:])
        else:
            conformal = ConformalRegressionPlotter(self.modelo_carregado, self.x_treino, self.dados_predicao_futuro[-8:], self.y_treino, self.dados_futuro[-8:])
        
        y_pred_best_model, y_pis_best_model, _, _ = conformal.regressao_conformal()
        return y_pred_best_model, y_pis_best_model, _, _

    def criando_dataframe_predicoes(self) -> pd.DataFrame:
        
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
    
    def predicao_ultimo_periodo(self) -> Any:
        dados_predicao = self.criando_dataframe_predicoes()
        data = dados_predicao.index[-1].strftime('%Y-%m-%d')
        intervalo_lower = dados_predicao['intervalo_lower'][-1]
        intervalo_upper = dados_predicao['intervalo_upper'][-1]
        predicao_proximo_mes = dados_predicao['predicao'][-1]
        return data,intervalo_lower,intervalo_upper,predicao_proximo_mes

    def plotando_predicoes(self, save: bool=False, diretorio: Optional[str]=None):
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
