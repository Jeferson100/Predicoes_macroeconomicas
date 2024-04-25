install:
	#pip install --upgrade pip --user &&\
    pip install --upgrade pip &&\
    	pip install --user -r requirements.txt	
format:	
	black economic_brazil/coleta_dados/*.py  economic_brazil/processando_dados/*.py economic_brazil/visualizacoes_graficas/*.py tests/*.py

lint:
	pylint --disable=R,C economic_brazil/*.py tests/*.py
test:
	python -m pytest -vv --cov=tests/test_*.py

refactor: format lint		
		
all: install lint format test