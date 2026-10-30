PYTHON ?= python

.PHONY: install install-dev format format-check lint test test-all

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

format:
	$(PYTHON) -m ruff format economic_brazil tests scripts
	$(PYTHON) -m ruff check --fix economic_brazil tests scripts

format-check:
	$(PYTHON) -m ruff format --check economic_brazil tests scripts
	$(PYTHON) -m ruff check economic_brazil tests scripts

lint:
	$(PYTHON) -m pylint economic_brazil

test:
	$(PYTHON) -m pytest -m "not network and not slow" --timeout=120 \
		--cov=economic_brazil --cov-report=term-missing

test-all:
	$(PYTHON) -m pytest

all: format lint test
