import os
import sys
from typing import Any, Dict, Optional

from nemoguardrails.actions import action

from src.services.application.rag_service import rag_service

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


generator_service = rag_service.rest_generator_service


async def get_query_response(
    user_question: str, session_id: Optional[str], user_id: Optional[str]
) -> str:
    """
    Gửi truy vấn đến hệ thống RAG để sinh câu trả lời dựa trên ngữ cảnh.

    Args:
        user_question (str): Câu hỏi của người dùng.
        session_id (Optional[str]): ID của phiên hội thoại hiện tại.
        user_id (Optional[str]): ID người dùng (phục vụ theo dõi hoặc context).

    Returns:
        str: Câu trả lời được sinh ra từ generator_service.
    """
    history = rag_service.get_session_history(session_id)
    print("length of history is ", len(history))
    print("user_id is ", user_id)
    print("session_id is ", session_id)

    # Gọi tới backend sinh dữ liệu dựa trên truy vấn + lịch sử
    return await generator_service.generate_rest_api(
        user_question,
        history.copy(),  # Dùng bản sao để không ảnh hưởng đến lịch sử gốc
        session_id,
        user_id,
    )


@action(is_system_action=True)
async def user_query(context: Optional[Dict[str, Any]] = None) -> str:
    """
    Action hệ thống để xử lý câu hỏi người dùng bằng cách gọi RAG generator.

    Args:
        context (Optional[Dict[str, Any]]): Ngữ cảnh hiện tại gồm messages.

    Returns:
        str: Câu trả lời sinh ra từ RAG backend hoặc thông báo lỗi nếu thiếu dữ liệu.
    """
    if context is None:
        return "No context provided."

    messages = context.get("user_message", [])

    user_question = None
    session_id = None
    user_id = None

    for message in messages:
        if message.get("role") == "user":
            user_question = message.get("content")
        elif message.get("role") == "context":
            context_content = message.get("content", {})
            session_id = context_content.get("session_id")
            user_id = context_content.get("user_id")

    if not user_question:
        return "Could not find user message in the context."

    return await get_query_response(user_question, session_id, user_id)


# Optional: Check if you use model classification do guardrail, uncomment this
# @action(name="self_check_input")
# async def self_check_input(llm_task_manager, context: dict, llm):
#     """
#     Checks if the user input should be allowed.

#     This action uses the "self_check_input" task to determine if the user's message
#     complies with the company policy. It is designed to work with models that return a
#     safety score (float) rather than a simple "Yes/No".

#     Args:
#         llm_task_manager: The LLM task manager from NeMo Guardrails.
#         context: The context dictionary containing the user input.

#     Returns:
#         bool: True if the input is allowed, False otherwise.
#     """
#     messages = context.get("user_message", [])
#     user_question = None
#     session_id = None
#     user_id = None
#     for message in messages:
#         if message.get("role") == "user":
#             user_question = message.get("content")
#         elif message.get("role") == "context":
#             context_content = message.get("content", {})
#             session_id = context_content.get("session_id")
#             user_id = context_content.get("user_id")

#     # Call the LLM with the self_check_input prompt
#     prompt = llm_task_manager.render_task_prompt(
#         task="self_check_input",
#         context={"user_input": user_question},
#     )
#     result = await llm_call(llm, prompt)
#     # The model returns a score. Lower is better.
#     # We assume scores < 0.5 are safe.
#     print("result", result)
#     score = float(result)
#     if score < 0.5:
#         return True  # Allowed

#     return False


# @action(name="self_check_output")
# async def self_check_output(llm_task_manager, context: dict, llm):
#     """
#     Checks if the user input should be allowed.

#     This action uses the "self_check_input" task to determine if the user's message
#     complies with the company policy. It is designed to work with models that return a
#     safety score (float) rather than a simple "Yes/No".

#     Args:
#         llm_task_manager: The LLM task manager from NeMo Guardrails.
#         context: The context dictionary containing the user input.

#     Returns:
#         bool: True if the input is allowed, False otherwise.
#     """
#     bot_response = context.get("bot_message")
#     if not bot_response:
#         return True  # Allow if there's no input to check

#     # Call the LLM with the self_check_input prompt
#     prompt = llm_task_manager.render_task_prompt(
#         task="self_check_output",
#         context={"bot_response": str(bot_response)},
#     )
#     result = await llm_call(llm, prompt)
#     # The model returns a score. Lower is better.
#     # We assume scores < 0.5 are safe.
#     score = float(result)
#     if score < 0.5:
#         return True  # Allowed

#     return False
