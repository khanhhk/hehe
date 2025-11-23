SHELL = /bin/bash
PYTHON := python3.11
VENV_NAME = rag_env
TEST_FOLDER = tests

# Environment
venv:
	${PYTHON} -m venv ${VENV_NAME} && \
	${VENV_NAME}/bin/pip install --upgrade pip && \
	${VENV_NAME}/bin/pip install -r requirements.txt && \
	${VENV_NAME}/bin/pre-commit install

# Style
style:
	black . --exclude ${VENV_NAME}
	flake8 . --exclude ${VENV_NAME}
	isort . --skip ${VENV_NAME}

test:
	@echo "🔍 Running flake8..."
	flake8 . --exclude=${VENV_NAME},.venv,__pycache__
	@echo "🔍 Running mypy..."
	mypy . --config-file mypy.ini
	@echo "✅ Running pytest..."
	CUDA_VISIBLE_DEVICES="" ${PYTHON} -m pytest -s --durations=0 --disable-warnings ${TEST_FOLDER}/
	@echo "🔍 Running pylint..."
	pylint . --ignore=${VENV_NAME},.venv,__pycache__ --recursive=y --output-format=colorized
.PHONY: venv style test
