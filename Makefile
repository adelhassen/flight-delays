SHELL := /bin/bash

env_vars:
	source set_env.sh

setup:
	pip install pre-commit
	pre-commit install
	pip install --upgrade pip
	pip install pipenv
	pipenv install --deploy

quality_check:
	black .

test:
	@python tests/unit_tests.py
	@python tests/integration_test.py
