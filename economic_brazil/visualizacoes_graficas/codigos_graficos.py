import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot, moment
import numpy as np
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
                plt.close()
            else:
                fig.show()
    
    def decomposicao_serie_temporal(self, dados, save=False, diretorio=None):
        for i in dados.columns:
            components = tsa.seasonal_decompose(dados[i], model='additive')
            ts = (dados[i].to_frame('Original')
                .assign(Trend=components.trend)
                .assign(Seasonality=components.seasonal)
                .assign(Residual=components.resid))

            with sns.axes_style('white'):
                ts.plot(subplots=True, figsize=(14, 8), title=['Original Series', 'Trend Component', 'Seasonal Component', 'Residuals'], legend=False)
                plt.suptitle(f'Seasonal Decomposition of {i}', fontsize=14)
                
                if save:
                    if diretorio is not None:
                        plt.savefig(diretorio + "/decomposicao_residuos_" + i + ".png")
                    plt.close()  # Isso evita que a figura seja exibida após salvar
                else:
                    plt.show()  # Exibe o gráfico apenas se não for salvar

                sns.despine()
                plt.tight_layout()
                plt.subplots_adjust(top=.91)
                
    def plot_correlogram(self, x, lags=None, title=None, save=False, diretorio=None):    
        lags = min(10, int(len(x)/5)) if lags is None else lags
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
        x.plot(ax=axes[0][0], title='Residuals')
        x.rolling(21).mean().plot(ax=axes[0][0], c='k', lw=1)
        q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
        stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
        axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
        probplot(x, plot=axes[0][1])
        mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
        s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
        axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
        plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
        plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
        axes[1][0].set_xlabel('Lag')
        axes[1][1].set_xlabel('Lag')
        fig.suptitle(title, fontsize=14)
        if save:
            plt.savefig(diretorio)
            plt.close() 
        else:
            plt.show()
                    
        sns.despine()
        
    def plot_correlogram_colunas(self,dados, lags_correlogram=None,  save_correlogram=False, diretorio_correlogram=None):
        for col in dados.columns:
            if save_correlogram:
                self.plot_correlogram(dados[col],lags=lags_correlogram,title=f'Correlogram_{col}', save=save_correlogram, diretorio=diretorio_correlogram+f"/correlogram_{col}.png")
            else:
                self.plot_correlogram(dados[col],lags=lags_correlogram,title=f'Correlogram_{col}')
    
    def plotar_residuos_predicit(self, y_treino, predict, bins=50, lags=40, save=False, diretorio=None):
        residuo = y_treino - predict
        plt.hist(residuo, bins=bins)
        plt.show()
        sm.graphics.tsa.plot_acf(residuo, lags=lags)
        if save:
            plt.savefig(diretorio)
            plt.close() 
        else:
            plt.show()
        