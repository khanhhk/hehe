from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from src.config.settings import SETTINGS
from src.utils.logger import FrameworkLogger, get_logger

logger: FrameworkLogger = get_logger()


class SummarizeService:
    """
    Service for summarizing long chat history using an LLM,
    reducing the number of tokens by summarizing old messages.
    """

    def __init__(self, langfuse_handler: CallbackHandler):
        """
        Initialize the summarization service.

        Args:
            langfuse_handler (CallbackHandler): Langfuse callback for tracing
            LLM interactions.
        """
        self.langfuse = get_client()
        self.llm = ChatOpenAI(**SETTINGS.llm_config)
        self.langfuse_handler = langfuse_handler

    async def _summarize_and_truncate_history(
        self,
        chat_history: list[dict],
        keep_last: int = 4,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> list[dict]:
        """
        Summarize the oldest portion of the chat history, then merge the summary
        with the remaining recent messages.

        Args:
            chat_history (list[dict]): Full chat history with `role` and `content`.
            keep_last (int): Number of most recent messages to retain.
            session_id (str | None): Optional Langfuse session ID.
            user_id (str | None): Optional Langfuse user ID.

        Returns:
            list[dict]: Truncated and summarized chat history.
        """
        if len(chat_history) <= keep_last:
            return chat_history

        try:
            # Split history: summarize the older portion
            old_messages = chat_history[:keep_last]
            remaining_messages = chat_history[keep_last:]

            # Build string for summarization prompt
            old_conversation = "\n".join(
                [
                    f"{msg['role'].capitalize()}: {msg['content']}"
                    for msg in old_messages
                ]
            )

            summary_prompt = (
                "Summarize this conversation in English, keeping key information "
                "(in 2-3 sentences):\n"
                f"{old_conversation}"
            )

            # Call LLM to summarize the old messages
            summary_msg = await self.llm.ainvoke(
                summary_prompt,
                {
                    "callbacks": [self.langfuse_handler],
                    "metadata": {
                        "langfuse_session_id": session_id,
                        "langfuse_user_id": user_id,
                    },
                },
            )

            # Return summarized message as a system message + recent ones
            summarized_history = [
                {
                    "role": "system",
                    "content": f"Previous conversation summary: {summary_msg.content}",
                }
            ] + remaining_messages

            logger.info(
                f"Summarized {len(old_messages)} old messages, "
                f"kept {len(remaining_messages)} recent messages"
            )
            return summarized_history

        except Exception as e:
            logger.error(f"Error summarizing history: {e}")
            # Fallback: only return the most recent messages
            return chat_history[-keep_last:]
