import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.io as pio
from economic_brazil.treinamento.redes_neurais_recorrentes import RnnModel
from typing import Optional, Any


class MetricasModelos:
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        algorithm: str,
        dados: Optional[pd.DataFrame] = None,
        save: bool = False,
        diretorio: Optional[str] = None,
    ) -> pd.DataFrame:
        # Calculando métricas básicas
        MAE = mean_absolute_error(y_true, y_pred)
        MSE = mean_squared_error(y_true, y_pred)
        RMSE = np.sqrt(MSE)
        R2 = r2_score(y_true, y_pred)
        variancia_y = np.var(y_pred)

        # Criando um dicionário de métricas com arredondamento
        metrics = {
            "MAE": round(float(MAE), 2),
            "MSE": round(float(MSE), 2),
            "RMSE": round(float(RMSE), 2),
            "R²": round(float(R2), 2),
            "Variance": round(variancia_y, 2),
        }

        # Criando um DataFrame a partir das métricas
        metrics_df = pd.DataFrame([metrics], index=[algorithm])

        # Se dados já existem, concatena, senão, retorna o novo DataFrame
        if dados is not None:
            metrics_df = pd.concat([dados, metrics_df])
            if save:
                metrics_df.to_csv(diretorio)
            return metrics_df

        else:
            return metrics_df

    def predicoes_comparando(
        self,
        dados_predicao,
        coluna,
        data_frame: Optional[pd.DataFrame] = None,
        index: Optional[pd.Index] = None,
        save: bool = False,
        diretorio: Optional[str] = None,
    ) -> pd.DataFrame:
        # Verifica se o data_frame não foi fornecido e então cria um novo
        if data_frame is None:
            data_frame = pd.DataFrame(index=index)
            data_frame[coluna] = dados_predicao
        else:
            # Adiciona a coluna ao DataFrame existente
            data_frame[coluna] = dados_predicao
            if save:
                data_frame.to_csv(diretorio)
        return data_frame

    def plotando_predicoes(
        self,
        dados: pd.DataFrame,
        title: str = "Predições",
        xlabel: str = "Tempo",
        ylabel: str = "Valores",
        figsize: tuple = (15, 10),
        grid: bool = True,
        save: bool = False,
        diretorio: Optional[str] = None,
    ) -> None:
        # Plotando as predicoes
        plt.figure(figsize=figsize)
        for i in dados.columns:
            plt.plot(dados.index, dados[i], label=i)

        plt.title(title)  # Adicionando o título do gráfico
        plt.xlabel(xlabel)  # Adicionando o rótulo do eixo x
        plt.ylabel(ylabel)  # Adicionando o rótulo do eixo y
        plt.legend()  # Mostrando a legenda
        plt.grid(grid)  # Adicionando grade ao gráfico para melhor visualização
        if save:
            plt.savefig(diretorio)
        else:
            plt.show()  # Exibindo o gráfico

    def plotando_predicoes_go(
        self,
        dados: pd.DataFrame,
        titulo: str = "Predições",
        label_x: str = "Tempo",
        label_y: str = "Valores",
        legenda: str = "predicoes",
        largura: int = 1000,
        altura: int = 600,
        model: str = "lines",
        save: bool = False,
        diretorio: Optional[str] = None,
    ):
        # Cria a figura para adicionar os traços (linhas do gráfico)
        fig = go.Figure()

        # Adiciona um traço para cada coluna no DataFrame
        for coluna in dados.columns:
            fig.add_trace(
                go.Scatter(x=dados.index, y=dados[coluna], mode=model, name=coluna)
            )

        # Adiciona títulos e ajusta o layout
        fig.update_layout(
            title=titulo,
            xaxis_title=label_x,
            yaxis_title=label_y,
            legend_title=legenda,
            width=largura,  # Define a largura da imagem
            height=altura,
        )

        # Exibe o gráfico
        if save:
            if diretorio is not None:
                if diretorio.endswith(".html"):
                    fig.write_html(diretorio)
                elif diretorio.endswith((".png", ".jpeg", ".jpg", ".svg", ".pdf")):
                    fig.write_image(diretorio)
                else:
                    print(
                        "Formato de arquivo não suportado. Use .html, .png, .jpeg, .jpg, .svg ou .pdf."
                    )
            else:
                print("Diretório não fornecido para salvar o gráfico.")
        else:
            fig.show()


class PredicaosModelos:
    def __init__(
        self,
        modelos: Any,
        x_treino: pd.DataFrame,
        y_treino: pd.DataFrame,
        x_teste: pd.DataFrame,
        y_teste: pd.DataFrame,
    ) -> None:
        self.modelos = modelos
        self.x_treino = x_treino
        self.y_treino = y_treino
        self.x_teste = x_teste
        self.y_teste = y_teste
        self.x_treino_recorrente, self.y_treino_recorrente = RnnModel().create_dataset(
            self.x_treino, self.y_treino
        )
        self.x_teste_recorrente, self.y_teste_recorrente = RnnModel().create_dataset(
            self.x_teste, self.y_teste
        )

    def predicoes(
        self,
        gradiente_boosting=True,
        xg_boost=True,
        cat_boost=True,
        regressao_linear=True,
        redes_neurais=True,
        sarimax=False,
        predicao_futuro=False,
        dados_predicao=None,
        dados_predicao_recorrente=None,
    ):
        if predicao_futuro:
            predicao_futuro = {}
            if gradiente_boosting:
                predicao_futuro["gradiente_boosting"] = self.modelos[
                    "gradiente_boosting"
                ].predict(dados_predicao)
            if xg_boost:
                predicao_futuro["xg_boost"] = self.modelos["xg_boost"].predict(
                    dados_predicao
                )
            if cat_boost:
                predicao_futuro["cat_boost"] = self.modelos["cat_boost"].predict(
                    dados_predicao
                )
            if regressao_linear:
                predicao_futuro["regressao_linear"] = self.modelos[
                    "regressao_linear"
                ].predict(dados_predicao)
            if redes_neurais:
                predicao_futuro["redes_neurais"] = (
                    self.modelos["redes_neurais"]
                    .predict(dados_predicao_recorrente)
                    .squeeze()
                )
            if sarimax:
                if dados_predicao is not None:
                    predicao_futuro["sarimax"] = self.modelos["sarimax"].predict(
                        start=0, end=dados_predicao.shape[0], exog=dados_predicao
                    )
                else:
                    print("Dados de predição não fornecidos para SARIMAX.")
            return predicao_futuro
        else:
            predicoes_treino = {}
            predicoes_teste = {}
            if gradiente_boosting:
                predicoes_treino["gradiente_boosting"] = self.modelos[
                    "gradiente_boosting"
                ].predict(self.x_treino)
                predicoes_teste["gradiente_boosting"] = self.modelos[
                    "gradiente_boosting"
                ].predict(self.x_teste)
            if xg_boost:
                predicoes_treino["xg_boost"] = self.modelos["xg_boost"].predict(
                    self.x_treino
                )
                predicoes_teste["xg_boost"] = self.modelos["xg_boost"].predict(
                    self.x_teste
                )
            if cat_boost:
                predicoes_treino["cat_boost"] = self.modelos["cat_boost"].predict(
                    self.x_treino
                )
                predicoes_teste["cat_boost"] = self.modelos["cat_boost"].predict(
                    self.x_teste
                )
            if regressao_linear:
                predicoes_treino["regressao_linear"] = self.modelos[
                    "regressao_linear"
                ].predict(self.x_treino)
                predicoes_teste["regressao_linear"] = self.modelos[
                    "regressao_linear"
                ].predict(self.x_teste)
            if redes_neurais:
                predicoes_treino["redes_neurais"] = (
                    self.modelos["redes_neurais"]
                    .predict(self.x_treino_recorrente)
                    .squeeze()
                )
                predicoes_teste["redes_neurais"] = (
                    self.modelos["redes_neurais"]
                    .predict(self.x_teste_recorrente)
                    .squeeze()
                )
            if sarimax:
                predicoes_treino["sarimax"] = self.modelos["sarimax"].predict(
                    start=0, end=self.x_treino.shape[0] - 2, exog=self.x_treino
                )
                predicoes_teste["sarimax"] = self.modelos["sarimax"].predict(
                    start=0, end=self.x_teste.shape[0] - 1, exog=self.x_teste
                )
            return predicoes_treino, predicoes_teste

    def return_dados(self):
        return (
            self.x_treino,
            self.y_treino,
            self.x_teste,
            self.y_teste,
            self.x_treino_recorrente,
            self.y_treino_recorrente,
            self.x_teste_recorrente,
            self.y_teste_recorrente,
        )


class MetricasModelosDicionario:
    def calculando_metricas(
        self, predicoes: dict, y: np.ndarray, y_recorrente: np.ndarray
    ) -> pd.DataFrame:
        metricas = MetricasModelos()
        inte = 0
        resultados = None
        for k, _ in predicoes.items():
            if inte == 0:
                resultados = metricas.evaluate_regression(y, predicoes[k], k)
            else:
                if k == "redes_neurais":
                    resultados = metricas.evaluate_regression(
                        y_recorrente, predicoes[k], k, dados=resultados
                    )
                else:
                    resultados = metricas.evaluate_regression(
                        y, predicoes[k], k, dados=resultados
                    )
            inte = inte + 1
        if resultados is not None:
            resultados = resultados.sort_values(by="MAE")
        return resultados if resultados is not None else pd.DataFrame()

    def plotando_predicoes(
        self,
        y_dados: np.ndarray,
        dados: dict,
        index: pd.Index,
        title: str = "Predições",
        xlabel: str = "Tempo",
        ylabel: str = "Valores",
        figsize: tuple = (15, 10),
        grid: bool = True,
        save: bool = False,
        diretorio: Optional[str] = None,
    ) -> None:
        # Plotando as predicoes
        plt.figure(figsize=figsize)
        plt.plot(index, y_dados, label="y")
        for i in dados.keys():
            if i == "redes_neurais":
                plt.plot(index[1:], dados[i], label=i)
            else:
                plt.plot(index, dados[i], label=i)

        plt.title(title)  # Adicionando o título do gráfico
        plt.xlabel(xlabel)  # Adicionando o rótulo do eixo x
        plt.ylabel(ylabel)  # Adicionando o rótulo do eixo y
        plt.legend()  # Mostrando a legenda
        plt.grid(grid)  # Adicionando grade ao gráfico para melhor visualização
        if save:
            plt.savefig(diretorio)
            plt.close()
        else:
            plt.show()  # Exibindo o gráfico

    def plotando_predicoes_treino_teste(
        self,
        y_dados_treino,
        y_dados_teste,
        predicao_treino,
        predicao_teste,
        index_treino,
        index_teste,
        title="Predições",
        xlabel="Tempo",
        ylabel="Valores",
        figsize=(15, 10),
        grid=True,
        save=False,
        diretorio=None,
    ):
        # Plotando as predicoes
        plt.figure(figsize=figsize)
        plt.plot(index_treino, y_dados_treino, label="y_treino")
        plt.plot(index_teste, y_dados_teste, label="y_teste")
        for i in predicao_treino.keys():
            if i == "redes_neurais":
                plt.plot(index_treino[1:], predicao_treino[i], label=f"{i}_treino")
            else:
                plt.plot(index_treino, predicao_treino[i], label=f"{i}_treino")
        for i in predicao_teste.keys():
            if i == "redes_neurais":
                plt.plot(index_teste[1:], predicao_teste[i], label=f"{i}_teste")
            else:
                plt.plot(index_teste, predicao_teste[i], label=f"{i}_teste")

        plt.title(title)  # Adicionando o título do gráfico
        plt.xlabel(xlabel)  # Adicionando o rótulo do eixo x
        plt.ylabel(ylabel)  # Adicionando o rótulo do eixo y
        plt.legend()  # Mostrando a legenda
        plt.grid(grid)  # Adicionando grade ao gráfico para melhor visualização
        if save:
            plt.savefig(diretorio)
            plt.close()
        else:
            plt.show()  # Exibindo o gráfico

    def plotando_predicoes_go_treino_teste(
        self,
        y_dados_treino: np.ndarray,
        y_dados_teste: np.ndarray,
        predicao_treino: dict,
        predicao_teste: dict,
        index_treino: pd.Index,
        index_teste: pd.Index,
        title: str = "Predições",
        xlabel: str = "Tempo",
        ylabel: str = "Valores",
        figsize: tuple = (15, 10),
        grid: bool = True,
        save: bool = False,
        diretorio: Optional[str] = None,
        type_arquivo: str = "png",
    ) -> None:
        """
        Função para plotar previsões de séries temporais usando Plotly.

        Parâmetros:
        y_dados_treino (array-like): Dados reais de treino.
        y_dados_teste (array-like): Dados reais de teste.
        predicao_treino (dict): Previsões dos modelos para os dados de treino.
        predicao_teste (dict): Previsões dos modelos para os dados de teste.
        index_treino (array-like): Índices para os dados de treino.
        index_teste (array-like): Índices para os dados de teste.
        title (str): Título do gráfico.
        xlabel (str): Rótulo do eixo x.
        ylabel (str): Rótulo do eixo y.
        figsize (tuple): Tamanho da figura do gráfico.
        grid (bool): Se True, adiciona uma grade ao gráfico.
        save (bool): Se True, salva o gráfico no diretório especificado.
        diretorio (str): Caminho do diretório para salvar o gráfico, se save for True.
        """
        fig = go.Figure()

        # Plotando os dados reais de treino e teste
        fig.add_trace(
            go.Scatter(
                x=index_treino,
                y=y_dados_treino,
                mode="lines",
                name="Dados de Treino",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_teste,
                y=y_dados_teste,
                mode="lines",
                name="Dados de Teste",
                line=dict(color="orange"),
            )
        )

        # Plotando as previsões de treino
        for modelo, pred in predicao_treino.items():
            label = f"{modelo} (Treino)"
            if modelo == "redes_neurais":
                fig.add_trace(
                    go.Scatter(
                        x=index_treino[1:],
                        y=pred,
                        mode="lines",
                        name=label,
                        line=dict(dash="dash"),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=index_treino,
                        y=pred,
                        mode="lines",
                        name=label,
                        line=dict(dash="dash"),
                    )
                )

        # Plotando as previsões de teste
        for modelo, pred in predicao_teste.items():
            label = f"{modelo} (Teste)"
            if modelo == "redes_neurais":
                fig.add_trace(
                    go.Scatter(
                        x=index_teste[1:],
                        y=pred,
                        mode="lines",
                        name=label,
                        line=dict(dash="dash"),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=index_teste,
                        y=pred,
                        mode="lines",
                        name=label,
                        line=dict(dash="dash"),
                    )
                )

        # Configurações do gráfico
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            autosize=False,
            width=figsize[0]
            * 100,  # Multiplicando por 100 para converter polegadas para pixels
            height=figsize[1] * 100,
            template="plotly_white" if grid else None,
        )

        # Salvando ou exibindo o gráfico
        if save:
            if diretorio:
                if type_arquivo == "png":
                    pio.write_image(fig, diretorio)
                if type_arquivo == "html":
                    pio.write_html(fig, file=diretorio, auto_open=False)
                else:
                    pio.write_image(fig, file=diretorio, format="svg")
            else:
                raise ValueError("Diretório não especificado para salvar o gráfico.")
        else:
            fig.show()
