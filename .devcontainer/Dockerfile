# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/codespaces-linux/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/universal:2-focal

COPY requirements.txt /tmp/

COPY Makefile /tmp/


RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get install -y python3-pip \  
     && pip3 install --no-cache-dir -r /tmp/requirements.txt


