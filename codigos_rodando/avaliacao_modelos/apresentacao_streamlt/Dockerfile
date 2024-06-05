## https://docs.streamlit.io/deploy/tutorials/docker

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Jeferson100/Predicoes_macroeconomicas.git

#WORKDIR /app/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/apresentacao_streamlt

RUN pip install --upgrade pip && \
    pip install -r /app/Predicoes_macroeconomicas/codigos_rodando/avaliacao_modelos/apresentacao_streamlt/requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_resultados.py", "--server.port=8501", "--server.address=0.0.0.0"]