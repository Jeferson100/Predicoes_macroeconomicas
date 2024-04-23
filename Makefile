install:
	#pip install --upgrade pip --user &&\
    pip install --upgrade pip &&\
    	pip install -r requirements.txt	
format:	
	black economic_brazil/*.py tests/*.py

lint:
	pylint --disable=R,C economic_brazil/*.py tests/*.py
test:
	python -m pytest -vv tests/test_*.py

refactor: format lint		
		
all: install lint format test