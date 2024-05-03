install:
	# Comando para atualizar pip e instalar dependências do arquivo requirements.txt
	# pip install --user -r requirements.txt
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black economic_brazil/coleta_dados/*.py economic_brazil/processando_dados/*.py economic_brazil/visualizacoes_graficas/*.py economic_brazil/treinamento/*.py economic_brazil/analisando_modelos/*.py tests/*.py

lint:
	pylint --disable=R,C economic_brazil/coleta_dados/*.py economic_brazil/processando_dados/*.py economic_brazil/visualizacoes_graficas/*.py economic_brazil/treinamento/*.py economic_brazil/analisando_modelos/*.py tests/*.py


test:
	python -m pytest -vv --cov=tests/test_*.py

refactor: format lint

all: install lint format test