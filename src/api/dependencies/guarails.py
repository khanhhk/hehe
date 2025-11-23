from fastapi import Request
from nemoguardrails import LLMRails


def get_guardrails_restapi(request: Request) -> LLMRails:
    """
    Dependency function to retrieve the REST API guardrails instance
    from the FastAPI app state.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        LLMRails: An instance of the LLMRails configured for REST API interaction.
    """
    return request.app.state.rails_restapi


def get_guardrails_sse(request: Request) -> LLMRails:
    """
    Dependency function to retrieve the SSE (Server-Sent Events)
    guardrails instance from the FastAPI app state.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        LLMRails: An instance of the LLMRails configured for SSE interaction.
    """
    return request.app.state.rails_sse
