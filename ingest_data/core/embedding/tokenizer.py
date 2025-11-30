from transformers import AutoTokenizer

from ingest_data.config import MODEL_NAME


def get_tokenizer():
    """
    Load and return the tokenizer from HuggingFace Transformers.

    Returns:
        transformers.PreTrainedTokenizer: Tokenizer corresponding to `MODEL_NAME`.
    """
    print(f"🔄 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer
