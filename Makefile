install:
	#pip install --upgrade pip &&\
	pip install --upgrade pip --user &&\
		pip install -r requirements.txt

#test:
#python -m pytest -vv --cov=main --cov=calCLI --cov=mylib test_*.py

format:	
	black Modelo_Previsao_Taxa_Selic/*.py

lint:
	pylint --disable=R,C Modelo_Previsao_Taxa_Selic/*.py

#container-lint:
#docker run --rm -i hadolint/hadolint < Dockerfile

refactor: format lint		
		
all: install lint format 