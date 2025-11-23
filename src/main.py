import argparse
import logging
import os
import tracemalloc
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from nemoguardrails import LLMRails, RailsConfig

from src.api.routers.api import api_router
from src.config.settings import APP_CONFIGS, SETTINGS
from src.services.application.rag_service import rag_service
from src.utils.logger import FrameworkLogger, get_logger

logger: FrameworkLogger = get_logger()
load_dotenv()
tracemalloc.start()


class EndpointFilter(logging.Filter):
    """
    Custom logging filter to ignore health and readiness probes in access logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return (
            record.args is not None
            and len(record.args) >= 3
            and list(record.args)[2] not in ["/health", "/ready"]
        )


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI to initialize services at startup.

    Initializes:
    - RAG service
    - Guardrails (REST & SSE)

    Args:
        app (FastAPI): The FastAPI app instance.
    """
    # Initialize RAG service
    fastapi_app.state.rag_service = rag_service

    # Load guardrails config for REST API
    config_restapi = RailsConfig.from_path(SETTINGS.GUARDRAILS_RESTAPI_PATH)
    fastapi_app.state.rails_restapi = LLMRails(config_restapi)

    # Load guardrails config for SSE
    config_sse = RailsConfig.from_path(SETTINGS.GUARDRAILS_SSE_PATH)
    fastapi_app.state.rails_sse = LLMRails(config_sse)

    yield


app = FastAPI(**APP_CONFIGS, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", include_in_schema=False)
async def healthcheck() -> dict[str, str]:
    """
    Health check endpoint for readiness probes.
    """
    return {"status": "ok"}


@app.get("/ready", include_in_schema=False)
async def readycheck() -> dict[str, str]:
    """
    Readiness check endpoint for container orchestration.
    """
    return {"status": "ok"}


app.include_router(
    api_router,
    prefix=SETTINGS.API_V1_STR,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG API server")
    parser.add_argument(
        "--provider",
        choices=["groq", "openai", "lm-studio"],
        required=True,
        help="LLM provider to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="environment_battery",
        help="Dataset name to use for RAG",
    )
    args = parser.parse_args()

    os.environ["LITELLM_MODEL"] = args.provider
    os.environ["DATASET_NAME"] = args.dataset

    logger.info("🚀 Starting RAG API server...")
    logger.info(f"   Provider: {args.provider}")
    logger.info(f"   Dataset: {args.dataset}")
    logger.info(f"   Collection: rag-pipeline-{args.dataset}")
    logger.info(f"   Host: {SETTINGS.HOST}:{SETTINGS.PORT}")
    logger.info(f"   API Docs: http://{SETTINGS.HOST}:{SETTINGS.PORT}/docs")

    uvicorn.run(
        "src.main:app",
        host=SETTINGS.HOST,
        port=SETTINGS.PORT,
        reload=True,
    )
