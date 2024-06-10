## https://docs.streamlit.io/deploy/tutorials/docker

##Para ativar o Streamlit, basta executar o seguinte comando: docker run -p 8501:8501 streamlit

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Jeferson100/Predicoes_macroeconomicas.git

WORKDIR /app/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/apresentacao_streamlit

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "1_Predicoes-Macroenomicas.py", "--server.port=8501", "--server.address=0.0.0.0"]