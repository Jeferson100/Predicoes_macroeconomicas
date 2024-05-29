import sys
sys.path.append("../../..")
from economic_brazil.coleta_dados.economic_data_brazil import data_economic
from economic_brazil.processando_dados.tratando_dados import TratandoDados
from economic_brazil.analisando_modelos.analise_modelos_regressao import MetricasModelosDicionario
from economic_brazil.analisando_modelos.analise_modelos_regressao import MetricasModelosDicionario,PredicaosModelos
from economic_brazil.treinamento.treinamento_algoritimos import carregar
from economic_brazil.analisando_modelos.regressao_conformal import ConformalRegressionPlotter
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title=f"Predicao Macroeconomicas: Variavel Selic",
                   page_icon="https://c.files.bbci.co.uk/1356A/production/_125801297_gettyimages-1218757425.jpg",
                   layout="wide")

#################################################### Carregando dados ############################################################

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
    modelos_carregados = carregar(diretorio='/workspaces/Predicoes_macroeconomicas/codigos_rodando/modelos_salvos/',gradiente_boosting=True, xg_boost=True, cat_boost=True, regressao_linear=True, redes_neurais=True, sarimax=False)
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

if 'melhor_modelo' not in st.session_state:
    melhor_modelo = metricas_teste['MAE'].idxmin()
    st.session_state['melhor_modelo'] = melhor_modelo
    
melhor_modelo = st.session_state['melhor_modelo']


if 'dados_conformal' not in st.session_state:
    dados_conformal = pd.read_csv('/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/regressao_conforma_teste.csv')
    st.session_state['dados_conformal'] = dados_conformal
dados_conformal = st.session_state['dados_conformal']

if 'dados_futuros' not in st.session_state:
    with open('/workspaces/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/dados_futuro.pkl', 'rb') as arquivo:
    # Carregar o objeto do arquivo
        st.session_state['dados_futuros'] =  pickle.load(arquivo)

dados_futuros = st.session_state['dados_futuros']


###################################################################### Titulo da Apresentação ########################################################
st.markdown(f"<h1 style='text-align: center; color: black;'>Predições Macrôeconomicas: Variável Selic para a data {dados_futuros['data']}</h1>", unsafe_allow_html=True)


###################################################### Sidebar github ########################################################

#Colocando um link para meu github
st.sidebar.markdown("Produzido por: https://github.com/Jeferson100")
# Create for Selecionador


###################################################### Filtrando modeloss ########################################################
st.sidebar.header("Escolha um dos Modelos:")

modelos_filtrados = st.sidebar.multiselect("Escolha um dos Modelos:", modelos_carregados.keys())

# Adcionando filtragem por modelos
modelos_filtrados_lista = []
for modelo in modelos_filtrados:
    modelos_filtrados_lista.append(modelo)

###################################################### Informacoes Futuras ########################################################################

def create_colored_box(text, value, color):
    return f"""
    <div style="
        background-color: {color}; 
        padding: 20px; 
        margin: 10px 0; 
        border-radius: 5px; 
        font-size: 20px; 
        text-align: center; 
        width: 100%; 
        margin-left: auto; 
        margin-right: auto;">
        <strong>{text}</strong>: {value}
    </div>
    """

st.markdown("## Informações da Previsão")
col1,col2,col3,col4 = st.columns(4)
with col1:
    st.markdown(create_colored_box("Data da Previsão", dados_futuros['data'], "#D3D3D3"), unsafe_allow_html=True)
with col2:
    st.markdown(create_colored_box("Previsão", np.round(dados_futuros['predicao'], 2), "#ADD8E6"), unsafe_allow_html=True)
with col3:
    st.markdown(create_colored_box("Intervalo Inferior", dados_futuros['intervalo_lower'], "#90EE90"), unsafe_allow_html=True)
with col4:
    st.markdown(create_colored_box("Intervalo Superior", dados_futuros['intervalo_upper'], "#FFB6C1"), unsafe_allow_html=True)
    
    
    
####################################################### Segundos graficos ########################################################################
col1,col2 = st.columns(2)
with col1:
    st.subheader("Predição dados de Teste")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index_teste, y=y_teste, mode='lines', name='Valores Reais'))
    if len(modelos_filtrados) > 0:
        for modelo in modelos_filtrados_lista:
            fig.add_trace(go.Scatter(x=index_teste, y=predicoes_teste[modelo], mode='lines', name=modelo))
    else:
        for modelo in predicoes_teste.keys():
            fig.add_trace(go.Scatter(x=index_teste, y=predicoes_teste[modelo], mode='lines', name=modelo))
        
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
    
    if len(modelos_filtrados) > 0:
        metricas_teste_filtrados = metricas_teste[metricas_teste.index.isin(modelos_filtrados_lista)]
    else:
        metricas_teste_filtrados = metricas_teste
        
    styled_metricas_teste = metricas_teste_filtrados.style\
        .applymap(lambda x: 'background-color: #f2f2f2' if x % 2 == 0 else '')\
        .set_properties(**{'text-align': 'center'})\
        .highlight_max(color='lightcoral')\
        .highlight_min(color='lightgreen')
    st.dataframe(styled_metricas_teste, use_container_width=True)
         
    # Exibir a tabela de métricas de treino no Streamlit
    st.subheader("Métricas de Treino")
    if len(modelos_filtrados) > 0:
        metricas_treino_filtrados = metricas_treino[metricas_treino.index.isin(modelos_filtrados_lista)]
    else:
        metricas_treino_filtrados = metricas_treino
    
    # Aplicar formatação para melhorar a visualização do DataFrame
    styled_metricas_treino = metricas_treino_filtrados.style\
        .applymap(lambda x: 'background-color: #f2f2f2' if x % 2 == 0 else '')\
        .set_properties(**{'text-align': 'center'})\
        .highlight_max(color='lightcoral')\
        .highlight_min(color='lightgreen')
        
    st.dataframe(styled_metricas_treino, use_container_width=True)   
    
###################################################### Terceiro graficos ########################################################################
col1,col2 = st.columns(2)
with col1:
    st.subheader("Predição dados de Treino e dados de Teste")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index_treino, y=y_treino, mode='lines', name='Valores Reais (Treino)'))
    if len(modelos_filtrados) > 0:
        for modelo in modelos_filtrados_lista:
            fig.add_trace(go.Scatter(x=index_treino, y=predicoes_treino[modelo], mode='lines', name=f'{modelo} (Treino)'))
    else:
        for modelo in predicoes_teste.keys():
            fig.add_trace(go.Scatter(x=index_treino, y=predicoes_treino[modelo], mode='lines', name=f'{modelo} (Treino)'))
    fig.add_trace(go.Scatter(x=index_teste, y=y_teste, mode='lines', name='Valores Reais (Teste)'))
    if len(modelos_filtrados) > 0:
        for modelo in modelos_filtrados_lista:
            fig.add_trace(go.Scatter(x=index_teste, y=predicoes_teste[modelo], mode='lines', name=f'{modelo} (Teste)'))
    else:
        for modelo in predicoes_teste.keys():
            fig.add_trace(go.Scatter(x=index_teste, y=predicoes_teste[modelo], mode='lines', name=f'{modelo} (Teste)'))
        
    fig.update_layout(
    
        xaxis_title="Tempo",
        yaxis_title="Selic",
        #legend_title="Legenda",
        height=600,
        width=800,
        yaxis=dict(range=[0, max(y_teste) * 2.2]),
        )
    st.plotly_chart(fig,use_container_width=True, height = 600, width = 2000)
    
    
with col2:
    
    st.subheader(f"Predição Conformal com o melhor modelo que foi {melhor_modelo.replace('_', ' ').title()}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index_treino, y=y_treino, mode='lines', name='Valores Reais Treino'))
    fig.add_trace(go.Scatter(x=index_teste, y=y_teste, mode='lines', name='Valores Reais Teste'))
    fig.add_trace(go.Scatter(x=index_teste, y=dados_conformal['predicao'], mode='lines', name=f'Predição Conformal'))
    fig.add_trace(
            go.Scatter(
                x=index_teste,
                y=dados_conformal['intervalo_lower'],
                fill=None,
                mode="lines",
                line=dict(color="lightgrey"),
                showlegend=False,
            )
        )
    fig.add_trace(
            go.Scatter(
                x=index_teste,
                y=dados_conformal['intervalo_upper'],
                fill="tonexty",
                mode="lines",
                line=dict(color="lightgrey"),
                showlegend=False,
            )
        )

    
        
    fig.update_layout(
    
        xaxis_title="Datas",
        yaxis_title="Selic",
        height=600,
        width=800,
        yaxis=dict(range=[0, max(y_teste) * 2]),
        )
    st.plotly_chart(fig,use_container_width=True, height = 600, width = 2000)
    
###################################################### Terceiro graficos ########################################################################





    


    
    




