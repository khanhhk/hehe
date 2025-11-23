from langchain_core.messages import SystemMessage
from langfuse import observe

from src.cache.semantic_cache import semantic_cache_llms
from src.utils.logger import FrameworkLogger, get_logger
from src.utils.text_processing import build_context

from .base import BaseGeneratorService

logger: FrameworkLogger = get_logger()


class RestApiGeneratorService(BaseGeneratorService):
    """
    Generator service for handling REST API-based RAG inference flows.

    This class manages a multi-phase pipeline:
    1. Initial LLM call (with optional tool call detection)
    2. Tool invocation (if applicable)
    3. Final RAG-based response generation
    """

    @observe(name="initial_llm_call_rest_api")
    async def _initial_llm_call(
        self,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        """
        Phase 1: Perform initial LLM call to detect tool calls (if any).

        Returns:
            - ai_msg: LLM response (may include tool_calls)
            - messages: SystemMessage-based input prompt to the LLM
        """
        self._update_trace_context(session_id, user_id)

        formatted_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in chat_history
        )

        prompt_template = self.prompt_userinput.get_langchain_prompt(
            question=question, chat_history=formatted_history
        )

        messages = [SystemMessage(content=prompt_template)]

        ai_msg = await self.llm_with_tools.ainvoke(
            messages,
            {
                "callbacks": [self.langfuse_handler],
                "metadata": {  # trace attributes
                    "langfuse_session_id": session_id,
                    "langfuse_user_id": user_id,
                },
            },
        )
        return ai_msg, messages

    @observe(name="create_message_rest_api")
    async def _create_message(
        self,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        """
        Phase 1 & 2: Initial LLM call → Tool execution (if tools exist).

        Returns:
            Tuple:
                - has_tools (bool): Whether tools were invoked.
                - str | list: Answer or updated message list.
        """
        # Phase 1: Initial LLM call with chat history
        ai_msg, messages = await self._initial_llm_call(
            question, chat_history, session_id, user_id
        )

        messages.append(ai_msg)

        # Kiểm tra tool calls
        tool_calls = ai_msg.additional_kwargs.get("tool_calls", [])

        if not tool_calls:
            # No tool calls — return the answer directly
            answer = self.clear_think.sub("", ai_msg.content).strip()
            return False, answer

        # Phase 2: Tool execution
        messages = await self._execute_tools(tool_calls, messages, session_id, user_id)

        return True, messages

    @observe(name="rag_generation_rest_api")
    @semantic_cache_llms.cache(namespace="post-cache")
    async def _rag_generation(
        self,
        messages: list,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        """
        Phase 3: Perform final RAG generation using tools + retrieved context.

        This method is cached via Redis Semantic Cache.
        """
        self._update_trace_context(session_id, user_id)

        context_str = build_context(messages)

        # RAG prompt với context
        prompt = self.prompt_rag.get_langchain_prompt(
            chat_history="\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in chat_history
            ),
            question=question,
            context=context_str,
        )

        # Final LLM call - không cần callbacks vì đã có built-in
        raw = await self.llm_with_tools.ainvoke(
            prompt,
            {
                "callbacks": [self.langfuse_handler],
                "metadata": {
                    "langfuse_session_id": session_id,
                    "langfuse_user_id": user_id,
                },
            },
        )

        content = raw.content if isinstance(raw.content, str) else str(raw.content)
        answer = self.clear_think.sub("", content).strip()

        return answer

    @observe(name="generate_rest_api")
    async def generate_rest_api(
        self,
        question: str,
        chat_history: list[dict],
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        """
        Entry point for serving REST API inference requests.

        Full flow:
            1. Initial LLM call (tool detection)
            2. Tool execution if needed
            3. Final RAG answer generation

        Returns:
            str: Final answer from the model.
        """
        try:
            has_tools, result = await self._create_message(
                question, chat_history, session_id, user_id
            )

            if not has_tools:
                return result  # Direct answer from LLM

            # Có tools - tiếp tục với RAG prompt
            messages = result
            answer = await self._rag_generation(
                messages=messages,
                question=question,
                chat_history=chat_history,
                session_id=session_id,
                user_id=user_id,
            )

            return answer

        except Exception as e:
            logger.error(f"Error in generate_rest_api(): {e}")
            raise
