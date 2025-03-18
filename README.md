
# 📈 Previsões Macroeconômicas da Economia Brasileira 

[![Teste Actions](https://github.com/Jeferson100/Predicoes_macroeconomicas/actions/workflows/teste.yml/badge.svg)](https://github.com/Jeferson100/Predicoes_macroeconomicas/actions/workflows/teste.yml)
[![Coleta de Dados](https://github.com/Jeferson100/Predicoes_macroeconomicas/actions/workflows/dados.yml/badge.svg)](https://github.com/Jeferson100/Predicoes_macroeconomicas/actions/workflows/dados.yml)
[![Treinando Modelos e Avaliando Modelos](https://github.com/Jeferson100/Predicoes_macroeconomicas/actions/workflows/treinando_avaliando_modelos.yml/badge.svg)](https://github.com/Jeferson100/Predicoes_macroeconomicas/actions/workflows/treinando_avaliando_modelos.yml)
[![Construindo and Push Docker Image](https://github.com/Jeferson100/Predicoes_macroeconomicas/actions/workflows/register_docker_streamlit.yml/badge.svg)](https://github.com/Jeferson100/Predicoes_macroeconomicas/actions/workflows/register_docker_streamlit.yml)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predicao-selic.streamlit.app/)


## 🎯 Introdução

Este projeto é uma introdução na **análise e previsão macroeconômica**, com foco especial na **economia brasileira**. Aqui, utilizamos o poder dos **dados históricos** e dos **modelos de Machine Learning** para tentar prever as tendências passadas e projetar o futuro de indicadores econômicos, como:

*   📊 **Produto Interno Bruto (PIB)**
*   📈 **Taxa de Inflação**
*   💼 **Taxa de Desemprego**
*   💵 **Taxa de Juros**
*   🔄 **Taxa de Câmbio**
*   ➕ **Muitos outros!**

## ✨ Motivação

A economia do Brasil é um organismo dinâmico, influenciado por varios fatores. Desde as políticas governamentais até as oscilações nos preços das commodities, cada evento causa um efeito em nossa economia. Diante desse cenário, **prever** os indicadores econômicos é uma necessidade:

*   **Investidores:** Para tomar decisões estratégicas e maximizar seus retornos.
*   **Tomadores de Decisão Políticos:** Para criar políticas eficazes e promover o crescimento econômico.
*   **Empresas:** Para se adaptarem às mudanças e prosperarem em um ambiente incerto.

## 🗂️ Dados Utilizados

Nossas análises são construídas sobre uma base de **dados macroeconômicos** provenientes de fontes diversas:

*   **🇧🇷 Instituto Brasileiro de Geografia e Estatística (IBGE)**
*   **🏦 Banco Central do Brasil**
*   **📊 Fundação Getúlio Vargas (FGV)**
* **📈 Ipeadata**
* **🌎 FRED(Federal Reserve Economic Data)**
* **👨‍💻 Google Trends**
*   **➕ Diversas outras fontes confiáveis**

Esses dados abrangem **décadas de informações**, detalhando as variáveis econômicas que moldam o cenário brasileiro.


## 🛠️ Metodologia

Aqui, a **ciência de dados** e o **Machine Learning** se encontram! Empregamos técnicas para construir modelos capazes de:

*   🕰️ **Analisar Séries Temporais:** Identificar padrões históricos e tendências.
*   🤖 **Aplicar Aprendizado de Máquina:** Criar modelos preditivos robustos e confiáveis.
*   🔄 **Ajustes Regulares:** Refinar os modelos com as informações econômicas mais recentes.

## 📊 Resultados e Acessibilidade

Este repositório é o coração do nosso trabalho, onde você encontrará:

*   **📈 Gráficos Interativos:** Para uma visualização clara das previsões e tendências.
*   **🧮 Análises Estatísticas:** Relatórios detalhados para uma compreensão profunda dos dados.
*   **📑 Relatórios Completos:** Um olhar aprofundado sobre os modelos e metodologias utilizadas.
* **👨‍💻 Streamlit App**: Para interagir com os dados e fazer sua propria analise

As previsões são atualizadas **regularmente**, refletindo as novidades do cenário econômico e os novos lançamentos de dados.

## 📂 Estrutura do Projeto: Detalhando os Módulos

O projeto é organizado em módulos distintos, cada um responsável por uma etapa específica do processo de análise e previsão.

### `economic_brazil/` (Módulo Principal)

Este é o diretório principal do projeto, que contém todos os submódulos relacionados à análise da economia brasileira.

#### `coleta_dados/` (Coleta de Dados)

*   **Responsabilidade:** Este módulo é responsável por coletar dados macroeconômicos de diversas fontes.
*   **Componentes Principais:**
    *   `economic_data_brazil.py`: Contém a classe `EconomicBrazil`, que orquestra a coleta de dados das seguintes fontes:
        *   **Banco Central do Brasil:** Dados de juros, câmbio e outros.
        *   **IBGE:** Dados de produção industrial, inflação, e mais.
        *   **Ipeadata:** Uma variedade de indicadores econômicos.
        *   **Google Trends:** Dados de tendências de pesquisa.
        *   **FRED:** Dados econômicos dos Estados Unidos.
    * **Funcionalidade:** Combinar dados dessas fontes em um dataframe unico
*   **Funcionalidade:** Reunir informações cruciais para as análises subsequentes, permitindo uma visão abrangente do cenário econômico.

#### `processando_dados/` (Processamento de Dados)

*   **Responsabilidade:** Este módulo lida com o tratamento, limpeza e transformação dos dados brutos coletados.
*   **Componentes Principais:**
    *   `data_processing.py`: Funções auxiliares para processar os dados, como criar dummies de COVID, calcular defasagens, criar colunas de mes e escalonar os dados.
    *   `estacionaridade.py`: A classe `Estacionaridade` implementa testes para verificar a estacionariedade das séries temporais (`test_kpss_adf`, `report_ndiffs`) e métodos para corrigir a não-estacionariedade (`corrigindo_nao_estacionaridade`).
    * `divisao_treino_teste.py`: funcoes para dividir o dataset em treino e teste.
    *   `tratando_dados.py`: A classe `TratandoDados` é o coração do processamento. Ela encapsula uma série de operações, como:
        *   Criação de dummies de COVID (`tratando_covid`).
        *   Correção da não-estacionariedade das séries (`tratando_estacionaridade`).
        *   Criação de colunas de mês, trimestre e dummies para datas (`tratando_datas`).
        *   Criação de defasagens (lags) (`tratando_defasagens`).
        *   Divisão dos dados em variáveis independentes (X) e dependentes (y) (`tratando_divisao_x_y`).
        *   Escalonamento dos dados usando `StandardScaler` (`tratando_scaler`).
        *   Aplicação de Análise de Componentes Principais (PCA) (`tratando_pca`).
        *   Aplicação de Recursive Feature Elimination (RFE) (`tratando_RFE`).
        *   Remoção de colunas com baixa variância (`tratando_variancia`).
        *   Remoção de colunas com alta correlação (`tratando_smart_correlation`).
        * Aplicação da diferenciação logaritmica (`diferenciacao_log`)
        * Divisão de dados em treino e teste (`tratando_divisao`)
        * Preparação dos dados para predição (`dados_futuros`)
        * Execução de todas as etapas acima em ordem (`tratando_dados`)
*   **Funcionalidade:** Transformar os dados brutos em um formato adequado para análise e modelagem, garantindo a qualidade e a relevância das informações.

#### `visualizacoes_graficas/` (Visualizações Gráficas)

*   **Responsabilidade:** Este módulo é dedicado à criação de visualizações gráficas que ajudam a entender os dados e os resultados dos modelos.
*   **Componentes Principais:**
    *   `codigos_graficos.py`: Contém a classe `Graficos`, que oferece diversas funções para gerar gráficos:
        *   `plotar_temporal`: Plota séries temporais.
        *   `plotar_residuos`: Plota histogramas e autocorrelação de resíduos.
        *   `plot_predict`: Plota previsões vs. dados reais.
        *   `plotar_heatmap`: Plota mapas de calor de correlação.
        *   `plotar_histograma`: Plota histogramas de variáveis.
        *   `go_plotar`: Plota gráficos interativos usando Plotly.
        *   `decomposicao_serie_temporal`: Plota a decomposição de uma série temporal (tendência, sazonalidade, resíduos).
        *   `plot_correlogram`: Plota correlogramas (ACF e PACF).
        * `plot_correlogram_colunas`: Plota correlogramas para varias colunas
        * `plotar_residuos_predicit`: plota histograma e autocorrelação dos residuos
*   **Funcionalidade:** Fornecer insights visuais sobre os dados, as relações entre as variáveis e os resultados das previsões, facilitando a interpretação.


## 🤝 Como Contribuir

Se você tem interesse em colaborar com o projeto, sinta-se à vontade. Sua colaboração é **valiosa**! Você pode nos ajudar a:

*   🚀 **Melhorar os Modelos:** Otimizar os algoritmos e aumentar a precisão.
*   ➕ **Adicionar Novas Fontes de Dados:** Enriquecer a análise com informações adicionais.
*   🐞 **Corrigir Erros:** Aprimorar a qualidade do código e dos resultados.
*   💡 **Propor Novas Análises:** Expandir o escopo do projeto e explorar novas ideias.

**Vamos melhorar as predicoes da economia brasileira!**

## 📞 Contatos

| GitHub | LinkedIn |
|--------|---------|
| [![GitHub](https://img.shields.io/badge/github-100000?style=for-the-badge&logo=github)](https://github.com/Jeferson100/Agente-investimento) | [![LinkedIn](https://img.shields.io/badge/linkedin-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jefersonsehnem/) |


