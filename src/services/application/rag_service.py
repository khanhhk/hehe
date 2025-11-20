import json
from logging import getLogger

from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from nemoguardrails import LLMRails

from src.cache.semantic_cache import semantic_cache_llms
from src.config.settings import SETTINGS
from src.infrastructure.vector_stores.chroma_client import ChromaClientService
from src.schemas.domain.retrieval import SearchArgs
from src.services.domain.generator.rest_api import RestApiGeneratorService
from src.services.domain.generator.sse import SSEGeneratorService
from src.services.domain.summarize import SummarizeService
from src.utils.text_processing import is_guardrails_error

logger = getLogger(__name__)


class Rag:
    def __init__(self):
        self.llm = ChatOpenAI(**SETTINGS.llm_config)
        self.chroma_client = ChromaClientService()
        self.langfuse_handler = CallbackHandler()
        self.langfuse = get_client()

        # Không cần in-memory storage nữa vì sẽ lấy từ Langfuse
        # self.session_histories: dict[str, list[dict]] = {}

        # Define search tool
        self.search_tool = StructuredTool.from_function(
            name="search_docs",
            description=(
                "Retrieve documents from Chroma.\n"
                "Args:\n"
                "    query (str): the query.\n"
                "    top_k (int): the number of documents to retrieve.\n"
                "    with_score (bool): whether to include similarity scores.\n"
                "    metadata_filter (dict): filter by metadata.\n"
            ),
            func=self.chroma_client.retrieve_vector,
            args_schema=SearchArgs,
        )

        # Define tools dictionary
        self.tools = {"search_docs": self.search_tool}

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(list(self.tools.values()))

        # Initialize services
        self.rest_generator_service = RestApiGeneratorService(
            llm_with_tools=self.llm_with_tools,
            tools=self.tools,
            langfuse_handler=self.langfuse_handler,
        )
        self.sse_generator_service = SSEGeneratorService(
            llm_with_tools=self.llm_with_tools,
            tools=self.tools,
            langfuse_handler=self.langfuse_handler,
        )

        self.summarize_service = SummarizeService(
            langfuse_handler=self.langfuse_handler,
        )

    def get_session_history(self, session_id: str | None = None) -> list[dict]:
        """Lấy chat history từ Langfuse dựa trên cấu trúc trace thực tế."""
        if not session_id:
            return []

        try:
            # Lấy traces từ Langfuse
            traces_in_session = self.langfuse.api.trace.list(
                session_id=session_id, limit=100
            )

            # Sắp xếp theo thời gian
            sorted_traces = sorted(traces_in_session.data, key=lambda x: x.timestamp)
            chat_history = []

            for trace in sorted_traces:
                ai_answer = ""
                user_question = ""

                if isinstance(trace.output, str):
                    ai_answer = trace.output
                elif isinstance(trace.output, dict):
                    ai_answer = trace.output.get("content", "") or trace.output.get(
                        "response", ""
                    )

                if isinstance(trace.input, dict):
                    user_question = trace.input.get("question", "")

                if user_question and ai_answer:
                    chat_history.extend(
                        [
                            {"role": "user", "content": user_question},
                            {"role": "assistant", "content": ai_answer},
                        ]
                    )

            max_pairs = 6
            return chat_history[-(max_pairs * 2) :]

        except Exception as e:
            print(f"Error fetching chat history from Langfuse: {e}")
            return []

    @semantic_cache_llms.cache(namespace="pre-cache")
    async def get_response(
        self,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
        guardrails: LLMRails | None = None,
    ):
        with self.langfuse.start_as_current_span(
            name="get_restapi_response",
            input={"question": question, "session_id": session_id, "user_id": user_id},
        ) as span:
            self.langfuse.update_current_trace(session_id=session_id, user_id=user_id)

            chat_history = self.get_session_history(session_id)
            print("chat_history is ", chat_history)

            # ———— Nếu có Guardrails thì dùng nó ————
            if guardrails:
                messages = [
                    {
                        "role": "context",
                        "content": {"session_id": session_id, "user_id": user_id},
                    },
                    {"role": "user", "content": question},
                ]
                # Guardrails tự động chạy input→dialog→output rails
                # KHÔNG trace guardrails.generate_async để tránh lưu config phức tạp
                result = await guardrails.generate_async(prompt=messages)

                if is_guardrails_error(result):
                    blocked_response = (
                        "I'm sorry, but I cannot provide a response to that request. "
                        "The content was blocked by our safety guidelines."
                    )
                    span.update(output=blocked_response)
                    return blocked_response

                response = str(result)
                span.update(output=response)
                return response

            # ———— Fallback: chạy RAG thường ————
            rag_output = await self.rest_generator_service.generate(
                question=question,
                chat_history=chat_history,
                session_id=session_id,
                user_id=user_id,
            )

            span.update(output=rag_output)
            return rag_output

    # ----------------------------------------------SSE----------------------------------------------

    async def _check_input_guardrails(self, question, session_id, user_id, guardrails):
        messages = [
            {
                "role": "context",
                "content": {"session_id": session_id, "user_id": user_id},
            },
            {"role": "user", "content": question},
        ]
        result = await guardrails.generate_async(
            messages=messages, options={"rails": ["input"]}
        )
        for msg in result.response:
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content")
                default_block = "I'm sorry, I can't respond to that."
                if default_block in assistant_msg:
                    return None, True, assistant_msg
            if msg.get("role") == "user" and msg.get("content") != question:
                question = msg.get("content")
        return question, False, None

    async def _stream_guardrails_response(
        self, question, session_id, user_id, chat_history, span, guardrails
    ):
        messages = [
            {
                "role": "context",
                "content": {"session_id": session_id, "user_id": user_id},
            },
            {"role": "user", "content": question},
        ]
        full_response = ""
        is_blocked = False

        async def rag_token_generator():
            async for message in self.sse_generator_service.generate_stream(
                question=question,
                chat_history=chat_history.copy(),
                session_id=session_id,
                user_id=user_id,
            ):
                yield message

        async for chunk in guardrails.stream_async(
            messages=messages, generator=rag_token_generator()
        ):
            full_response += chunk
            if is_guardrails_error(chunk):
                is_blocked = True
                error_msg = (
                    "I'm sorry, but I cannot provide a response to that request."
                )
                yield f"{json.dumps(error_msg)}\n\n"
                break
            else:
                yield f"{json.dumps(chunk)}\n\n"

        span.update(
            output=full_response if not is_blocked else "Request blocked by guardrails"
        )

    async def _stream_plain_rag_response(
        self, question, session_id, user_id, chat_history, span
    ):
        full_response = ""
        async for message in self.sse_generator_service.generate_stream(
            question=question,
            chat_history=chat_history.copy(),
            session_id=session_id,
            user_id=user_id,
        ):
            full_response += message
            yield f"{json.dumps(message)}\n\n"
        span.update(output=full_response)

    @semantic_cache_llms.cache(namespace="pre-cache")
    async def get_sse_response(
        self,
        question: str,
        session_id: str,
        user_id: str,
        guardrails: LLMRails | None = None,
    ):
        with self.langfuse.start_as_current_span(
            name="get_sse_response",
            input={"question": question, "session_id": session_id, "user_id": user_id},
        ) as span:
            self.langfuse.update_current_trace(session_id=session_id, user_id=user_id)
            chat_history = self.get_session_history(session_id)

            # Step 1: Check input guardrails
            if guardrails:
                question, is_blocked, assistant_msg = (
                    await self._check_input_guardrails(
                        question, session_id, user_id, guardrails
                    )
                )
                if is_blocked:
                    yield f"{json.dumps(assistant_msg)}\n\n"
                    span.update(output="Request blocked by input guardrails")
                    return

                # Step 2: Stream with guardrails
                async for chunk in self._stream_guardrails_response(
                    question, session_id, user_id, chat_history, span, guardrails
                ):
                    yield chunk
                return

            # Step 3: Stream with plain RAG
            async for chunk in self._stream_plain_rag_response(
                question, session_id, user_id, chat_history, span
            ):
                yield chunk


rag_service = Rag()
