from enum import Enum


class LLMModel(str, Enum):
    """
    Enum representing supported LLM model identifiers.
    """

    OPENAI_TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    OPENAI_GPT_4O_MINI = "gpt-4o-mini"


class LLMProvider(str, Enum):
    """
    Enum representing supported LLM service providers.
    """

    OPENAI = "openai"
