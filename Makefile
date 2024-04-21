install:
	#pip install --upgrade pip --user &&\
    pip install --upgrade pip &&\
    	pip install -r requirements.txt	
format:	
	black Modelo_Previsao_Taxa_Selic/*.py

lint:
	pylint --disable=R,C Modelo_Previsao_Taxa_Selic/*.py
test:
	python -m pytest -vv tests_economics/test_*.py

refactor: format lint		
		
all: install lint format test