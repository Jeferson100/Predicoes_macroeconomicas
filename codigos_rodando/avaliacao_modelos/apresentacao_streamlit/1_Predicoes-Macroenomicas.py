import sys
sys.path.append("../../..")
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
warnings.filterwarnings("ignore")

st.set_page_config(page_title=f"Predições Macroeconomicas",
                   page_icon="https://c.files.bbci.co.uk/1356A/production/_125801297_gettyimages-1218757425.jpg",
                   layout="wide")


#dados_path = os.path.join(os.getcwd(), 'dados_salvos.pkl')
path_diretorio = os.getcwd()

#################################################### Carregando dados ############################################################
try:
    arquivo = path_diretorio+'/dados_salvos_selic.pkl'
    dados_salvos_selic = pickle.load(open(arquivo, 'rb'))
except FileNotFoundError:
    arquivo = '/mount/src/predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/apresentacao_streamlit/dados_salvos_selic.pkl'
    dados_salvos_selic = pickle.load(open(arquivo, 'rb'))
    
try:
    from economic_brazil.coleta_dados.economic_data_brazil import data_economic
    dados_economicos = data_economic()
except:
    sys.path.append('/mount/src/predicoes_macroeconomicas')
    from economic_brazil.coleta_dados.economic_data_brazil import data_economic
    dados_economicos = data_economic()
    #dados_economicos = pd.read_csv('/mount/src/Predicoes_macroeconomicas/dados/dados_bcb.csv')

if "dados_salvos_selic" not in st.session_state:
    st.session_state['dados_futuro'] = dados_salvos_selic['dados_futuro']
    
if "dados_economicos" not in st.session_state:
    st.session_state['dados_economicos'] = dados_economicos
   
dados_futuros_selic = st.session_state['dados_futuro']
dados_economicos = st.session_state['dados_economicos']



def juntar_dados(primeira_vez=True,recebe_data=None,dicionario_1=None, dicionario_2=None,variavel_1=None,variavel_2=None):
    if primeira_vez:
        data_1 = pd.DataFrame([dicionario_1])
        data_1['Variavel'] = variavel_1
        data_2 = pd.DataFrame([dicionario_2])
        data_2['Variavel'] = variavel_2
        return pd.concat([data_1,data_2])
    else:
        data_1 = pd.DataFrame([dicionario_1])
        data_1['Variavel'] = variavel_1
        return pd.concat([data_1,recebe_data])
    
predicoes = juntar_dados(primeira_vez=True,recebe_data=None,dicionario_1=dados_futuros_selic, dicionario_2=dados_futuros_selic,variavel_1='selic',variavel_2='selic_2')
predicoes.index = predicoes['Variavel']
data = predicoes['data'].iloc[0]
predicoes.drop(['Variavel','data'],axis=1,inplace=True)
predicoes = predicoes[['predicao', 'intervalo_upper', 'intervalo_lower']]


###################################################################### Titulo da Apresentação ########################################################
st.markdown(f"<h1 style='text-align: center; color: black;'>Predições Macrôeconomicas para a data {data}</h1>", unsafe_allow_html=True)


###################################################### Sidebar github ########################################################

st.sidebar.markdown(
    '''
    <img src="https://c.files.bbci.co.uk/1356A/production/_125801297_gettyimages-1218757425.jpg" 
    style="width: 300px; height: auto;">
    ''',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

###################################################### Filtrando modeloss ########################################################
st.sidebar.header("Variáveis Macroeconômicas:")

#filtro_variaveis = st.sidebar.multiselect("Escolha uma das Variáveis Macroeconômicas:", predicoes.index.unique())
filtro_variaveis = st.sidebar.selectbox("Escolha uma das Variáveis Macroeconômicas:", predicoes.index.unique())

st.sidebar.markdown("---")

# Exibir imagem do GitHub
st.sidebar.markdown("<h1 style='text-align: ; color: black;'>Contatos</h1>", unsafe_allow_html=True)

st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github)](https://github.com/Jeferson100)")


if filtro_variaveis:
    dados_economicos_filtrados =   dados_economicos[filtro_variaveis]
    predicoes_filtradas = predicoes[predicoes.index.unique() == filtro_variaveis]
else:
    dados_economicos_filtrados = dados_economicos['selic']
    predicoes_filtradas = predicoes[predicoes.index.unique() == 'selic']

    
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
if filtro_variaveis:
    st.markdown(f"## Previsão da Variável {filtro_variaveis}:")
else:
    st.markdown(f"## Previsão da Variável Selic:")
col1,col2,col3,col4 = st.columns(4)
with col1:
    st.markdown(create_colored_box("Data da Previsão", data, "#D3D3D3"), unsafe_allow_html=True)
with col2:
    st.markdown(create_colored_box("Previsão", np.round(predicoes_filtradas['predicao'].squeeze(), 2), "#ADD8E6"), unsafe_allow_html=True)
with col3:
    st.markdown(create_colored_box("Intervalo Inferior", predicoes_filtradas['intervalo_lower'].squeeze(), "#90EE90"), unsafe_allow_html=True)
with col4:
    st.markdown(create_colored_box("Intervalo Superior", predicoes_filtradas['intervalo_upper'].squeeze(), "#FFB6C1"), unsafe_allow_html=True)
    
####################################################### Primeiro grafico ########################################################################
if filtro_variaveis:
    st.subheader("Histórico da Variável "+filtro_variaveis)
else:   
    st.subheader("Histórico da Variável Selic")
fig = go.Figure()
fig.add_trace(go.Scatter(x=dados_economicos_filtrados.index, y=dados_economicos_filtrados, mode='lines', name=f'Histórico da Variável {filtro_variaveis}'))
fig.update_layout(
        title="",
        xaxis_title="Periódo",
        yaxis_title="Selic",
        legend_title="Legenda",
        height=1000,
        width=800,
        yaxis=dict(range=[0, max(dados_economicos_filtrados) * 1.1]),
        )
st.plotly_chart(fig,use_container_width=True, height = 1000, width = 2000)


####################################################### Tabela ########################################################################

st.subheader("Previsão de todas as Variáveis Macroeconômicas")
   
    # Aplicar formatação para melhorar a visualização do DataFrame

cores = ['#d9d9d9','#d9d9d9','#f2f2f2', '#d9d9d9', '#bfbfbf', '#a6a6a6', '#87CEEB', '#FFD700', '#90EE90']

def alternate_row_colors(row):
    return ['background-color: %s' % cores[idx % len(cores)] for idx in range(len(row))]

## colorindo as colunas https://discuss.streamlit.io/t/pandas-styler-with-custom-style/37039/10


header_styles = {
    'predicao': [{'selector': 'th', 'props': [('background-color', '#d9d9d9')] }],
    'intervalo_lower': [{'selector': 'th', 'props': [('background-color', '#d9d9d9')] }],
    'intervalo_upper': [{'selector': 'th', 'props': [('background-color', '#d9d9d9')] }],
}

# Função para aplicar cores alternadas nas linhas
styled_predicoes = predicoes.style\
    .apply(alternate_row_colors, axis=0)\
    .set_properties(**{'text-align': 'center'})\
    .set_table_styles(header_styles)\
    .set_properties(**{'text-align': 'center'})

#st.table(styled_predicoes)

st.dataframe(styled_predicoes,
             column_config={
                 "Variavel":  st.column_config.TextColumn("Variaveis",), 
                 "predicao": st.column_config.NumberColumn("Predição",), 
                 "intervalo_lower": st.column_config.NumberColumn("Intervalo Inferior",), 
                 "intervalo_upper": st.column_config.NumberColumn("Intervalo Superior",)}, 
             use_container_width=True)




