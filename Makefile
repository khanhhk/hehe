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
	flake8 . --exclude ${VENV_NAME}
	mypy . --exclude ${VENV_NAME}
	CUDA_VISIBLE_DEVICES="" ${PYTHON} -m pytest -s --durations=0 --disable-warnings ${TEST_FOLDER}/
	pylint . --ignore=${VENV_NAME}

.PHONY: venv style test
