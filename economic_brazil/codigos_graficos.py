import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")


def plotar(dados):
  for i in range(len(dados.columns)):
    fig, ax = plt.subplots(dpi=120)
    ax.plot(dados.iloc[:,i],label=dados.columns[i])
    plt.legend()
    plt.show()
    
    
def plotar_residuos(y_treino,predict,bins=50, lags=40):
    residuo = y_treino - predict
    plt.hist(residuo,bins=bins)
    plt.show()
    sm.graphics.tsa.plot_acf(residuo,lags=lags)
    plt.show()
    
def plot_predict(predict,y_teste):
    fig, ax = plt.subplots(dpi=120)
    ax.plot(y_teste,label="y_teste")
    ax.plot(predict,label="predict")
    plt.legend()
    plt.show()
    