import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")


class Graficos:
    def plotar_temporal(self, dados, save=False, diretorio=None):
        for i in range(len(dados.columns)):
            _, ax = plt.subplots(dpi=120)
            ax.plot(dados.iloc[:, i], label=dados.columns[i])
            plt.legend()
            if save:
                plt.savefig(diretorio + "/" + dados.columns[i] + ".png")
            else:
                plt.show()

    def plotar_residuos(
        self, y_treino, predict, bins=50, lags=40, save=False, diretorio=None
    ):
        residuo = y_treino - predict
        plt.hist(residuo, bins=bins)
        plt.show()
        sm.graphics.tsa.plot_acf(residuo, lags=lags)
        if save:
            plt.savefig(diretorio)
        else:
            plt.show()

    def plot_predict(self, y_teste, predict, save=False, diretorio=None):
        _, ax = plt.subplots(dpi=120)
        ax.plot(y_teste, label="y_teste")
        ax.plot(predict, label="predict")
        plt.legend()
        if save:
            plt.savefig(diretorio)
        else:
            plt.show()

    def plotar_heatmap(self, dados, save=False, diretorio=None, size=(15, 10)):
        sns.set_theme(rc={"figure.figsize": size})
        sns.heatmap(dados.corr(), cmap="YlGnBu", annot=True)
        if save:
            plt.savefig(diretorio)
        else:
            plt.show()

    def plotar_histograma(self, dados, save=False, diretorio=None):
        for i in range(len(dados.columns)):
            sns.histplot(i, kde=True)
            if save:
                plt.savefig(diretorio)
            else:
                plt.show()

    def go_plotar(self, dados, save=False, diretorio=None):
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
            if save:
                fig.write_html(diretorio)
            else:
                fig.show()
