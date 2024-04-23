import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")


class Graficos:
  def __init__(self, dados, y_treino, predict, y_teste):
      self.dados = dados
      self.y_treino = y_treino
      self.predict = predict
      self.y_teste = y_teste
    
  def plotar_temporal(self):
      for i in range(len(self.dados.columns)):
          fig, ax = plt.subplots(dpi=120)
          ax.plot(self.dados.iloc[:,i],label=self.dados.columns[i])
          plt.legend()
          plt.show()
               
  def plotar_residuos(self,bins=50, lags=40):
      residuo = self.y_treino - self.predict
      plt.hist(residuo,bins=bins)
      plt.show()
      sm.graphics.tsa.plot_acf(residuo,lags=lags)
      plt.show()
          
  def plot_predict(self):
      fig, ax = plt.subplots(dpi=120)
      ax.plot(self.y_teste,label="y_teste")
      ax.plot(self.predict,label="predict")
      plt.legend()
      plt.show()

  def plotar_heatmap(self):
      sns.set_theme(rc={'figure.figsize':(15,10)})
      sns.heatmap(self.dados.corr(), cmap="YlGnBu", annot=True)
      plt.show()

  def plotar_histograma(self):
      for i in range(len(self.dados.columns)):
          sns.histplot(i, kde=True)
          plt.show()

  def go_plotar(self):
    fig = go.Figure()
    for i in range(len(self.dados.columns)):
       fig.add_trace(go.Scatter(x=self.dados.index, y=self.dados.iloc[:,i], name=self.dados.columns[i]))
       fig.show()

