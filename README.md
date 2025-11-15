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
