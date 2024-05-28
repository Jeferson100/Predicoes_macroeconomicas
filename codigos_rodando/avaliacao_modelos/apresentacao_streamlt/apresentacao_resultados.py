import sys
sys.path.append("../../..")
from economic_brazil.coleta_dados.economic_data_brazil import data_economic
from economic_brazil.processando_dados.tratando_dados import TratandoDados
from economic_brazil.analisando_modelos.analise_modelos_regressao import MetricasModelosDicionario
from economic_brazil.analisando_modelos.analise_modelos_regressao import MetricasModelosDicionario,PredicaosModelos
from economic_brazil.treinamento.treinamento_algoritimos import carregar
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title=f"Predicao Macroeconomicas: Variavel Selic",
                   page_icon="https://c.files.bbci.co.uk/1356A/production/_125801297_gettyimages-1218757425.jpg",
                   layout="wide")

#Titulo
st.title(f"Predições Macroeconomicas: Variavel Selic")

if "dados" not in st.session_state:
    dados = data_economic()
    st.session_state["dados"] = dados
    
    
dados = st.session_state["dados"]

if 'x_treino' not in st.session_state:
    tratando = TratandoDados(dados)
    x_treino, x_teste, y_treino, y_teste,pca, scaler = tratando.tratando_dados()
    data_divisao_treino_teste = tratando.data_divisao_treino_teste()
    st.session_state['x_treino'] = x_treino
    st.session_state['x_teste'] = x_teste
    st.session_state['y_treino'] = y_treino
    st.session_state['y_teste'] = y_teste
    st.session_state['pca'] = pca
    st.session_state['scaler'] = scaler
    st.session_state['data_divisao_treino_teste'] = data_divisao_treino_teste
    
x_treino = st.session_state['x_treino']
x_teste = st.session_state['x_teste']
y_treino = st.session_state['y_treino']
y_teste = st.session_state['y_teste']
pca = st.session_state['pca']
scaler = st.session_state['scaler']
data_divisao_treino_teste = st.session_state['data_divisao_treino_teste']

if 'index_treino' not in st.session_state:
    index_treino = dados[dados.index <= data_divisao_treino_teste][6:].index
    st.session_state['index_treino'] = index_treino
    index_teste = dados[dados.index > data_divisao_treino_teste][4:].index
    st.session_state['index_teste'] = index_teste

index_treino = st.session_state['index_treino']
index_teste = st.session_state['index_teste']

if 'modelos_carregados' not in st.session_state:
    modelos_carregados = carregar(diretorio='/workspaces/Predicoes_macroeconomicas/codigos_rodando/modelos_salvos/',gradiente_boosting=True, xg_boost=True, cat_boost=True, regressao_linear=True, redes_neurais=True, sarimax=True)
    st.session_state['modelos_carregados'] = modelos_carregados
modelos_carregados = st.session_state['modelos_carregados']

predi = PredicaosModelos(modelos_carregados, x_treino, y_treino, x_teste, y_teste)

if 'y_teste_recorrente' not in st.session_state:
    predicoes_treino, predicoes_teste = predi.predicoes()
    x_treino, y_treino, x_teste, y_teste, x_treino_recorrente, y_treino_recorrente, x_teste_recorrente, y_teste_recorrente = predi.return_dados()
    st.session_state['y_teste_recorrente'] = y_teste_recorrente
    st.session_state['y_treino_recorrente'] = y_treino_recorrente
    st.session_state['x_teste_recorrente'] = x_teste_recorrente
    st.session_state['x_treino_recorrente'] = x_treino_recorrente
    
y_teste_recorrente = st.session_state['y_teste_recorrente']
y_treino_recorrente = st.session_state['y_treino_recorrente']
x_teste_recorrente = st.session_state['x_teste_recorrente']
x_treino_recorrente = st.session_state['x_treino_recorrente']   
    
if 'metricas_teste' not in st.session_state:
    metri = MetricasModelosDicionario()
    predicoes_treino, predicoes_teste = predi.predicoes()
    metrica_teste = metri.calculando_metricas(predicoes_teste, y_teste, y_teste_recorrente)
    metrica_treino = metri.calculando_metricas(predicoes_treino, y_treino, y_treino_recorrente)
    st.session_state['predicoes_treino'] = predicoes_treino
    st.session_state['predicoes_teste'] = predicoes_teste
    st.session_state['metricas_teste'] = metrica_teste
    st.session_state['metricas_treino'] = metrica_treino

predicoes_treino = st.session_state['predicoes_treino']
predicoes_teste = st.session_state['predicoes_teste']
metricas_teste = st.session_state['metricas_teste']
metricas_treino = st.session_state['metricas_treino']


# Create for Selecionador
st.sidebar.header("Escolha um dos Modelos:")

#Colocando um link para meu github
st.sidebar.markdown("Produzido por: https://github.com/Jeferson100")

#metricas_treino = pd.read_csv('/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/metricas_treino.csv')
metricas_teste =pd.read_csv('/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/metricas_teste.csv',index_col=0)

col1,col2 = st.columns(2)
with col1:
    st.subheader("Predição dados de Teste")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index_teste, y=y_teste, mode='lines', name='Valores Reais'))
    if 'redes_neurais' in predicoes_teste.keys():
        fig.add_trace(go.Scatter(x=index_teste, y=predicoes_teste['redes_neurais'], mode='lines', name='Redes Neurais'))
    if 'regressao_linear' in predicoes_teste.keys():
        fig.add_trace(go.Scatter(x=index_teste, y=predicoes_teste['regressao_linear'], mode='lines', name='Regressão Linear'))
    if 'xg_boost' in predicoes_teste.keys():
        fig.add_trace(go.Scatter(x=index_teste, y=predicoes_teste['xg_boost'], mode='lines', name='XG Boost'))
    if 'cat_boost' in predicoes_teste.keys():
        fig.add_trace(go.Scatter(x=index_teste, y=predicoes_teste['cat_boost'], mode='lines', name='Cat Boost'))
    if 'gradiente_boosting' in predicoes_teste.keys():
        fig.add_trace(go.Scatter(x=index_teste, y=predicoes_teste['gradiente_boosting'], mode='lines', name='Gradiente Boosting'))
    if 'sarimax' in predicoes_teste.keys():
        fig.add_trace(go.Scatter(x=index_teste, y=predicoes_teste['sarimax'], mode='lines', name='SARIMAX'))
    fig.update_layout(
        title="Predição dados de Teste",
        xaxis_title="Tempo",
        yaxis_title="Selic",
        legend_title="Legenda",
        height=600,
        width=800,
        yaxis=dict(range=[0, max(y_teste) * 1.4]),
        )
    st.plotly_chart(fig,use_container_width=True, height = 600, width = 2000)
  


"""with col2:
    # Exibir a tabela no Streamlit
    st.subheader("Métricas de Teste")
    st.dataframe(metricas_teste, use_container_width=True)
    st.subheader("Métricas de Treino")
    st.dataframe(metricas_treino, use_container_width=True)
    #st.subheader('Métricas de Teste')
    #st.write(metricas_teste)"""

with col2:
    # Exibir a tabela de métricas de teste no Streamlit
    st.subheader("Métricas de Teste")
    
    # Aplicar formatação para melhorar a visualização do DataFrame
    styled_metricas_teste = metricas_teste.style\
        .applymap(lambda x: 'background-color: #f2f2f2' if x % 2 == 0 else '')\
        .set_properties(**{'text-align': 'center'})\
        .highlight_max(color='lightcoral')\
        .highlight_min(color='lightgreen')
    st.dataframe(styled_metricas_teste, use_container_width=True)
         
    # Exibir a tabela de métricas de treino no Streamlit
    st.subheader("Métricas de Treino")
    
    # Aplicar formatação para melhorar a visualização do DataFrame
    styled_metricas_treino = metricas_treino.style\
        .applymap(lambda x: 'background-color: #f2f2f2' if x % 2 == 0 else '')\
        .set_properties(**{'text-align': 'center'})\
        .highlight_max(color='lightcoral')\
        .highlight_min(color='lightgreen')
        
    st.dataframe(styled_metricas_treino, use_container_width=True)   
    
    






