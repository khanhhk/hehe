import uuid

from fastapi import APIRouter, Depends, status
from nemoguardrails import LLMRails

from src.api.dependencies.guarails import get_guardrails_restapi
from src.api.dependencies.rag_dependency import get_rag_service
from src.schemas.api.requests import UserInput
from src.schemas.api.response import ResponseOutput
from src.services.application.rag_service import Rag
from src.utils.logger import FrameworkLogger, get_logger

logger: FrameworkLogger = get_logger()
router = APIRouter()


@router.post(
    "/",
    status_code=status.HTTP_200_OK,
    response_model=ResponseOutput,
)
async def retrieve_restaurants(
    user_input: UserInput,
    rag_service: Rag = Depends(get_rag_service),
    guardrails: LLMRails = Depends(get_guardrails_restapi),
) -> ResponseOutput:
    """
    REST endpoint for generating a single RAG response with optional
    Guardrails filtering.

    This route:
    - Accepts a user question (`user_input`) and optional identifiers.
    - Ensures consistent `session_id` and `user_id` (generating them if missing).
    - Delegates response generation to the RAG service.
    - Returns the structured output wrapped in a `ResponseOutput` model.

    Args:
        user_input (UserInput): User request payload containing `user_input`,
        `session_id`, and `user_id`.
        rag_service (Rag): Dependency-injected RAG service responsible for
        orchestrating LLM + retrieval.
        guardrails (LLMRails): Optional Guardrails safety/validation layer (REST mode).

    Returns:
        ResponseOutput: A structured response including model output, session ID
        and user ID.
    """
    logger.info(f"You are in rest api {user_input.session_id}-{user_input.user_id}")
    # ———— ID Normalization ————
    session_id = user_input.session_id or str(uuid.uuid4())
    user_id = user_input.user_id or f"user_{uuid.uuid4().hex[:8]}"
    response = await rag_service.get_response(
        question=user_input.user_input,
        session_id=session_id,
        user_id=user_id,
        guardrails=guardrails,
    )

    return ResponseOutput(
        response=response,
        session_id=session_id,
        user_id=user_id,
    )
