from typing import List, Union

from langchain_core.messages import BaseMessage, ToolMessage


def build_context(messages: List[BaseMessage]) -> str:
    """
    Build a context string from a list of messages by extracting
    the content of ToolMessage instances.

    Args:
        messages (List[BaseMessage]): A list of BaseMessage instances,
        potentially including ToolMessages.

    Returns:
        str: A string containing the concatenated content from all
        ToolMessages, separated by '\n\n--- Retrieved Documents ---\n\n'.
    """
    tool_chunks = []
    for m in messages:
        # Only extract content from ToolMessage instances
        if isinstance(m, ToolMessage):
            tool_chunks.append(str(m.content))

    # Join all extracted contents with a separator indicating document boundaries
    context_str = "\n\n--- Retrieved Documents ---\n\n".join(tool_chunks)
    return context_str


def is_guardrails_error(response: Union[str, dict]) -> bool:
    """ "
    Check if a given response (from REST API or SSE) indicates a guardrails
    violation or blocked content.

    Args:
        response (Union[str, dict]): The response to be checked, could be a
        dict (REST) or string (SSE).

    Returns:
        bool: True if a guardrails error or blocking message is
        detected, False otherwise.
    """

    # If response is a dictionary (use for restapi)
    if isinstance(response, dict):
        # Check for direct error key
        if "error" in response:
            return True

    # If response is a string (use for sse)
    response_str = str(response)

    # Keywords or phrases commonly indicating a guardrails violation or blocking
    error_indicators = [
        "guardrails_violation",
        "Blocked by self check output rails",
        "content_blocked",
        "I'm sorry, I can't respond to that",
        '"error":',
        "blocked by guardrails",
    ]

    # Check if any indicator exists in the response string (case-insensitive)
    return any(
        indicator.lower() in response_str.lower() for indicator in error_indicators
    )
