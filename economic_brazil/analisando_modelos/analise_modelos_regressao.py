import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

class MetricasModelos:
    def evaluate_regression(self,y_true, y_pred, algorithm, dados=None,save=None, diretorio=None):
        # Calculando métricas básicas
        MAE = mean_absolute_error(y_true, y_pred)
        MSE = mean_squared_error(y_true, y_pred)
        RMSE = np.sqrt(MSE)
        R2 = r2_score(y_true, y_pred)

        # Criando um dicionário de métricas com arredondamento
        metrics = {
            'MAE': round(MAE, 2),
            'MSE': round(MSE, 2),
            'RMSE': round(RMSE, 2),
            'R²': round(R2, 2)
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

    def predicoes_comparando(self,dados_predicao, coluna, data_frame: pd.DataFrame = None, index=None, save=False, diretorio=None):
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

    def plotando_predicoes(self,dados,title="Predições", xlabel="Tempo", ylabel="Valores", figsize=(15, 10),grid=True, save=False, diretorio=None):
        #Plotando as predicoes
        plt.figure(figsize=figsize)
        for i in dados.columns:
            plt.plot(dados.index,dados[i],label=i)
        
        plt.title(title)  # Adicionando o título do gráfico
        plt.xlabel(xlabel)  # Adicionando o rótulo do eixo x
        plt.ylabel(ylabel)  # Adicionando o rótulo do eixo y
        plt.legend()  # Mostrando a legenda
        plt.grid(grid)  # Adicionando grade ao gráfico para melhor visualização
        if save:
            plt.savefig(diretorio)
        else:  
            plt.show()  # Exibindo o gráfico
    
    def plotando_predicoes_go(self,dados, titulo='Predições', label_x='Tempo',label_y='Valores',legenda='predicoes', largura=1000, altura=600,model='lines',save=None,diretorio=None):
        # Cria a figura para adicionar os traços (linhas do gráfico)
        fig = go.Figure()

        # Adiciona um traço para cada coluna no DataFrame
        for coluna in dados.columns:
            fig.add_trace(go.Scatter(x=dados.index, y=dados[coluna], mode=model, name=coluna))

        # Adiciona títulos e ajusta o layout
        fig.update_layout(
            title=titulo,
            xaxis_title=label_x,
            yaxis_title=label_y,
            legend_title=legenda,
            width=largura,  # Define a largura da imagem
            height=altura
        )

        # Exibe o gráfico
        if save:
            # Salva o gráfico no caminho especificado
            if diretorio.endswith('.html'):
                fig.write_html(diretorio)
            elif diretorio.endswith(('.png', '.jpeg', '.jpg', '.svg', '.pdf')):
                fig.write_image(diretorio)
            else:
                print("Formato de arquivo não suportado. Use .html, .png, .jpeg, .jpg, .svg ou .pdf.")
        else:
            fig.show()
