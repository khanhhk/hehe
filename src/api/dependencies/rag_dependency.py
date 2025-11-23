from fastapi import Request

from src.services.application.rag_service import Rag


def get_rag_service(request: Request) -> Rag:
    """
    Dependency function to retrieve the Rag service instance from the FastAPI app state.

    This function is used with FastAPI's `Depends()` to inject the shared RAG service
    object into request handlers or other dependency functions.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        Rag: An instance of the Rag service stored in the application state.
    """
    return request.app.state.rag_service
