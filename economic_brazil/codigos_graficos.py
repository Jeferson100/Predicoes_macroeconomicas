import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")


class Graficos:
    def plotar_temporal(self, dados):
        for i in range(len(dados.columns)):
            _, ax = plt.subplots(dpi=120)
            ax.plot(dados.iloc[:, i], label=dados.columns[i])
            plt.legend()
            plt.show()

    def plotar_residuos(self, y_treino, predict, bins=50, lags=40):
        residuo = y_treino - predict
        plt.hist(residuo, bins=bins)
        plt.show()
        sm.graphics.tsa.plot_acf(residuo, lags=lags)
        plt.show()

    def plot_predict(self, y_teste, predict):
        _, ax = plt.subplots(dpi=120)
        ax.plot(y_teste, label="y_teste")
        ax.plot(predict, label="predict")
        plt.legend()
        plt.show()

    def plotar_heatmap(self, dados):
        sns.set_theme(rc={"figure.figsize": (15, 10)})
        sns.heatmap(dados.corr(), cmap="YlGnBu", annot=True)
        plt.show()

    def plotar_histograma(self, dados):
        for i in range(len(dados.columns)):
            sns.histplot(i, kde=True)
            plt.show()

    def go_plotar(self, dados):
        fig = go.Figure()
        for i in range(len(dados.columns)):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=dados.index, y=dados.iloc[:, i], name=dados.columns[i])
            )
            fig.update_layout(
                title=f"Grafico da variavel {dados.columns[i]}",
                xaxis_title="Anos",
                yaxis_title="Valores",
            )
            fig.show()
