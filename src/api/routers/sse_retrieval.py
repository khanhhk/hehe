import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from nemoguardrails import LLMRails

from src.api.dependencies.guarails import get_guardrails_sse
from src.api.dependencies.rag_dependency import get_rag_service
from src.schemas.api.requests import UserInput
from src.services.application.rag_service import Rag

router = APIRouter()


@router.post(
    "/",
    status_code=status.HTTP_200_OK,
)
async def retrieve_restaurants(
    user_input: UserInput,
    rag_service: Rag = Depends(get_rag_service),
    guardrails: LLMRails = Depends(get_guardrails_sse),
) -> StreamingResponse:
    """
    SSE endpoint for streaming RAG responses with guardrails integration.

    This endpoint:
    - Accepts a user question via `user_input`
        (which includes optional session/user IDs).
    - Ensures `session_id` and `user_id` are generated if not provided.
    - Sends an initial metadata event (with session/user info).
    - Streams the LLM response using Server-Sent Events (SSE),
        enforcing Guardrails if configured.

    Args:
        user_input (UserInput): The incoming user request payload, including input text
        and optional IDs.
        rag_service (Rag): The RAG service to generate responses
        (injected via dependency).
        guardrails (LLMRails): The guardrails runtime for filtering/moderating
        output (injected).

    Returns:
        StreamingResponse: An SSE response that streams chunks of generated text.
    """
    try:
        # Check và generate session_id/user_id ở router
        session_id = user_input.session_id or str(uuid.uuid4())
        user_id = user_input.user_id or f"user_{str(uuid.uuid4())[:8]}"

        async def generate_response():
            # Gửi metadata trước
            metadata = {"session_id": session_id, "user_id": user_id}
            yield f"metadata: {json.dumps(metadata)}\n\n"

            # Stream response
            async for chunk in rag_service.get_sse_response(
                question=user_input.user_input,
                session_id=session_id,
                user_id=user_id,
                guardrails=guardrails,
            ):
                yield chunk

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
        )
    except asyncio.TimeoutError:
        return StreamingResponse("responseUpdate: [Timeout reached.]")
