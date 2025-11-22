# RAG Chatbot System
## System Architecture
![](images/Architecture.svg)

## Quick Start
### Install Dependencies
```shell
make venv
``` 
### Run the Application
#### A. Start Infrastructure Services
Start the Redis cache service:
```shell
docker compose -f infrastructure/cache/docker-compose.yaml up -d
``` 

Start the Langfuse observability stack:
```shell
docker compose -f infrastructure/observability/docker-compose.yaml up -d
``` 

Start the data ingestion services (Airflow, Minio):
```shell
docker compose -f ingest_data/docker-compose.yaml up -d
``` 
#### B. Run the Data Ingestion Pipeline
#### C. Start the RAG API Server
```shell
# Use the default dataset (environment_battery)
python -m src.main --provider groq

# Or specify a different dataset that you've ingested
python -m src.main --provider groq --dataset llm_papers
``` 
The server will be accessible at `http://localhost:8000` by default. You can access the API documentation at `http://localhost:8000/docs`.

## Project Structure
```txt
├── .github
│   └── workflows
│       └── ci-cd.yaml
├── guardrails
│   ├── config_restapi
│   │   ├── rails
│   │   │   └── disallowed.co
│   │   ├── actions.py
│   │   ├── config.yml
│   │   └── prompts.yml
│   └── config_sse
│       ├── rails
│       │   └── disallowed.co
│       ├── config.yml
│       └── prompts.yml
├── infrastructure
│   ├── observability
│   │   └── docker-compose.yaml
│   ├── router
│   │   ├── config.yaml
│   │   ├── docker-compose.yml
│   │   └── example.env
│   └── docker-compose.yaml
├── ingest_data
│   ├── config
│   │   ├── airflow.cfg
│   │   └── config.yaml
│   ├── dags
│   │   └── ingesting_data.py
│   ├── plugins
│   │   ├── config
│   │   │   └── minio_config.py
│   │   ├── jobs
│   │   │   ├── __init__.py
│   │   │   ├── download.py
│   │   │   ├── embed_and_store.py
│   │   │   ├── load_and_chunk.py
│   │   │   └── utils.py
│   │   └── __init__.py
│   ├── .gitignore
│   ├── Dockerfile
│   ├── README.md
│   ├── __init__.py
│   ├── docker-compose.yaml
│   └── requirements.txt
├── src
│   ├── api
│   │   ├── dependencies
│   │   │   ├── guarails.py
│   │   │   └── rag_dependency.py
│   │   ├── routers
│   │   │   ├── __init__.py
│   │   │   ├── api.py
│   │   │   ├── rest_retrieval.py
│   │   │   └── sse_retrieval.py
│   │   └── __init__.py
│   ├── config
│   │   ├── prompts.yaml
│   │   └── settings.py
│   ├── constants
│   │   ├── enum.py
│   │   └── prompt.py
│   ├── infrastructure
│   │   ├── embeddings
│   │   │   └── embeddings.py
│   │   └── vector_stores
│   │       └── chroma_client.py
│   ├── schemas
│   │   ├── api
│   │   │   ├── requests.py
│   │   │   └── response.py
│   │   └── domain
│   │       └── retrieval.py
│   ├── services
│   │   ├── application
│   │   │   └── rag_service.py
│   │   └── domain
│   │       ├── generator
│   │       │   ├── base.py
│   │       │   ├── rest_api.py
│   │       │   └── sse.py
│   │       └── summarize.py
│   ├── utils
│   │   ├── logger.py
│   │   └── text_processing.py
│   ├── __init__.py
│   └── main.py
├── tests
│   └── test_base.py
├── .flake8
├── .gitignore
├── .isort.cfg
├── .pre-commit-config.yaml
├── .pylintrc
├── Dockerfile
├── Makefile
├── README.md
├── mypy.ini
└── requirements.txt
```


