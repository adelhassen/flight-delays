SHELL := /bin/bash

setup:
	pre-commit install
	source set_env.sh

quality_check:
	black .

test:
	@python tests/unit_tests.py
	@python tests/integration_test.py
