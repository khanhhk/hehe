import json
import re
from abc import ABC, abstractmethod
from typing import Optional

from langchain.tools import StructuredTool
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.runnables import Runnable
from langfuse import get_client
from langfuse.langchain import CallbackHandler


class BaseGeneratorService(ABC):
    """
    Abstract base class for generator services (REST or SSE).
    Integrates LangChain LLMs with structured tools and Langfuse tracing.

    Subclasses must implement the main logic for:
    - initial LLM call
    - structured message creation
    - RAG response generation
    """

    def __init__(
        self,
        llm_with_tools: Runnable[LanguageModelInput, BaseMessage],
        tools: dict[str, StructuredTool],
        langfuse_handler: CallbackHandler,
    ):
        """
        Initialize the generator service.

        Args:
            llm_with_tools: A LangChain Runnable that wraps
            the LLM + tool calling logic.
            tools: A dictionary mapping tool names to StructuredTool instances.
            langfuse_handler: A Langfuse callback handler for observability/tracing.
        """
        self.llm_with_tools = llm_with_tools
        self.tools = tools
        self.langfuse = get_client()
        self.prompt_userinput = self.langfuse.get_prompt(
            "userinput_service",
            label="production",
            type="text",
        )
        self.prompt_rag = self.langfuse.get_prompt(
            "rag_service",
            label="production",
            type="text",
        )
        self.clear_think = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
        self.langfuse_handler = langfuse_handler

    def _update_trace_context(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Update Langfuse trace context with session/user ID.
        """
        if session_id:
            self.langfuse.update_current_trace(session_id=session_id)
        if user_id:
            self.langfuse.update_current_trace(user_id=user_id)

    @abstractmethod
    async def _initial_llm_call(
        self,
        question: str,
        chat_history: list[dict],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Abstract method to perform the initial LLM call.
        Should return tool invocations or LLM messages.
        """
        pass

    @abstractmethod
    async def _create_message(
        self,
        question: str,
        chat_history: list[dict],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Abstract method to create formatted messages for the LLM.
        Should structure conversation context for downstream processing.
        """
        pass

    async def _execute_tools(
        self,
        tool_calls: list[dict],
        messages: list[BaseMessage],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> list[BaseMessage]:
        """
        Execute tool calls from the LLM and append ToolMessages to the message list.

        Args:
            tool_calls: Parsed list of tool calls from the LLM output.
            messages: Current message chain to append results to.
            session_id: Langfuse session identifier.
            user_id: Langfuse user identifier.

        Returns:
            Updated list of messages including tool outputs.
        """
        self._update_trace_context(session_id, user_id)

        for tool_call in tool_calls:
            name = tool_call["function"]["name"].lower()

            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")

            tool_inst = self.tools[name]
            payload = json.loads(tool_call["function"]["arguments"])

            with self.langfuse.start_as_current_span(
                name=f"tool_{name}", input=payload, metadata={"tool_name": name}
            ) as span:
                # Handle multi-call tools (tool_calls array inside payload)
                if "tool_calls" in payload:
                    for call_args in payload["tool_calls"]:
                        # Trace từng call args nếu có nhiều
                        with self.langfuse.start_as_current_span(
                            name=f"tool_{name}_call", input=call_args
                        ) as sub_span:
                            output = tool_inst.invoke(call_args)
                            sub_span.update(output=output)

                        messages.append(
                            ToolMessage(
                                content=output, tool_call_id=tool_call.get("id")
                            )
                        )
                else:
                    output = tool_inst.invoke(payload)
                    span.update(output=output)
                    messages.append(
                        ToolMessage(content=output, tool_call_id=tool_call.get("id"))
                    )

        return messages

    @abstractmethod
    async def _rag_generation(
        self,
        messages: list[BaseMessage],
        question: str,
        chat_history: list[dict],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Abstract method to generate a final RAG-based answer from the LLM.

        Args:
            messages: The complete conversation history in LangChain format.
            question: The original user query.
            chat_history: Raw chat history in dict format.
            session_id: Optional Langfuse session ID.
            user_id: Optional Langfuse user ID.
        """
        pass
