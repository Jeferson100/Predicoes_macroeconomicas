install:
	# Comando para atualizar pip e instalar dependências do arquivo requirements.txt
	# pip install --user -r requirements.txt
	pip install --upgrade pip && \
	pip install -r requirements.txt

uv_install:
	pip install uv && \
	uv pip install --upgrade pip && \
		uv pip install -r requirements.txt


format:
	black economic_brazil/coleta_dados/*.py economic_brazil/processando_dados/*.py economic_brazil/visualizacoes_graficas/*.py economic_brazil/treinamento/*.py economic_brazil/analisando_modelos/*.py tests/*.py

ruff_format:
	ruff format economic_brazil/coleta_dados/*.py economic_brazil/processando_dados/*.py economic_brazil/visualizacoes_graficas/*.py economic_brazil/treinamento/*.py economic_brazil/analisando_modelos/*.py tests/*.py

#lint:pylint --disable=R,C economic_brazil/coleta_dados/*.py economic_brazil/processando_dados/*.py economic_brazil/visualizacoes_graficas/*.py economic_brazil/treinamento/*.py economic_brazil/analisando_modelos/*.py tests/*.py

lint:
	call for /r "economic_brazil" %%f in (*.py) do pylint --disable=R,C "%%f" & call for /r "tests" %%f in (*.py) do pylint --disable=R,C "%%f"

ruff_lint:
	ruff check economic_brazil/coleta_dados/*.py economic_brazil/processando_dados/*.py economic_brazil/visualizacoes_graficas/*.py economic_brazil/treinamento/*.py economic_brazil/analisando_modelos/*.py tests/*.py

typepyright:
	pyright economic_brazil/coleta_dados/*.py economic_brazil/processando_dados/*.py economic_brazil/visualizacoes_graficas/*.py economic_brazil/treinamento/*.py economic_brazil/analisando_modelos/*.py tests/*.py

typemypy:
	mypy check economic_brazil/coleta_dados/ economic_brazil/processando_dados/ economic_brazil/visualizacoes_graficas/ economic_brazil/treinamento/ economic_brazil/analisando_modelos/

test:
	python -m pytest -vv --cov=tests/test_*.py

refactor: format lint

all: uv_install format lint typepyright typemypy ruff_format ruff_lint test