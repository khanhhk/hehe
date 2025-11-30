import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

from ingest_data.config import MODEL_NAME


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load and return a HuggingFace sentence embedding model using LangChain.

    Returns:
        HuggingFaceEmbeddings: Embedding model with device auto-selected.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device.upper()} ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME, model_kwargs={"device": device}
    )
    return embeddings
