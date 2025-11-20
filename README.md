```
src/
├── api/
|    ├── dependencies
|    |   ├── guarails.py
|    |   └── rag.py
|    └── routers
|    |   ├── api.py
|    |   ├── rest_retrieval.py
|    |   └── sse_retrieval.py
├── cache/
|    ├── semantic_cache.py
|    └── standard_cache.py
├── config/
|    ├── prompts.yaml
|    └── settings.py
├── constants/
|    ├── enum.py
|    └── prompt.py
├── infrastructure/
|    ├── embeddings
|    |   └── embeddings.py
|    └── vector_stores
|    |   └── chroma_client.py
├── schemas/
|    ├── api
|    |   ├── requests.py
|    |   └── response.py
|    └── domain
|    |   └── retrieval.py
├── services/
|    ├── application
|    |   └── rag.py
|    └── domain
|    |   ├── generator
|    |   |   ├── base.py
|    |   |   ├── rest_api.py
|    |   |   └── sse.py
|    |   └── summarize.py
├── utils/
|    ├── logger.py
|    └── text_processing.py
└── main.py
```

### Infrastructure Services
#### Start the Redis cache service:
docker compose -f infrastructure/cache/docker-compose.yaml up -d
#### Start the Langfuse observability stack:
docker compose -f infrastructure/observability/docker-compose.yaml up -d
#### Start the data ingestion services (Airflow, Minio):
docker compose -f ingest_data/docker-compose.yaml up -d

### Start the RAG API Server
python -m src.main --provider groq
