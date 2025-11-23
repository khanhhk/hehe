from typing import Any, Dict, List, Optional, Tuple

from langchain.schema.document import Document
from langchain_chroma import Chroma

from src.config.settings import SETTINGS
from src.infrastructure.embeddings.embeddings import EmbeddingService


def _format_docs(docs: List[Document], scores: Optional[List[float]] = None) -> str:
    """
    Format a list of LangChain Document objects into a readable string.

    Args:
        docs (List[Document]): List of retrieved documents.
        scores (Optional[List[float]]): Optional list of similarity scores.

    Returns:
        str: Formatted document contents, optionally with scores.
    """
    formatted = []
    for idx, doc in enumerate(docs):
        content = doc.page_content.strip()
        if scores:
            content += f" [score={scores[idx]:.4f}]"
        formatted.append(content)
    return "\n\n".join(formatted)


class ChromaClientService:
    """
    Wrapper class for interacting with a Chroma vector store using LangChain.

    Handles connection, document retrieval, and similarity search
    with optional filtering and scores.
    """

    def __init__(self) -> None:
        """
        Initialize the Chroma client service without establishing a connection yet.
        """
        self.embedding_service = EmbeddingService()
        self.client = Chroma(
            collection_name=SETTINGS.CHROMA_COLLECTION_NAME,
            persist_directory=SETTINGS.CHROMA_PERSIST_DIR,
            embedding_function=self.embedding_service,
        )

    def retrieve_vector(
        self,
        query: str,
        top_k: int = 3,
        with_score: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Perform a vector similarity search for a query string.

        Args:
            query (str): The search query.
            top_k (int): Number of top similar documents to retrieve.
            with_score (bool): Whether to include similarity scores in the result.
            metadata_filter (Optional[Dict[str, Any]]): Metadata filter to apply.

        Returns:
            str: Formatted string of the retrieved documents (with scores if requested).
        """

        if with_score:
            docs_with_scores: List[Tuple[Document, float]] = (
                self.client.similarity_search_with_score(
                    query, k=top_k, filter=metadata_filter
                )
            )
            try:
                retrieved_docs, scores = zip(*docs_with_scores)
                return _format_docs(list(retrieved_docs), list(scores))
            except ValueError:
                return "Không tìm thấy tài liệu phù hợp."

        else:
            docs: List[Document] = self.client.similarity_search(
                query, k=top_k, filter=metadata_filter
            )
            return _format_docs(docs)
