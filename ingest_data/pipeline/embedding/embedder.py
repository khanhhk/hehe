from uuid import uuid4

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from ingest_data.config import (
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MODEL_NAME,
)
from ingest_data.plugins.jobs.utils import MinioLoader, get_embeddings


class DocumentEmbedder:
    """
    Class for generating and storing document embeddings into a Chroma vector store.

    - Uses a multilingual sentence-transformers model to compute embeddings.
    - Supports metadata cleaning via LangChain utilities.
    - Stores resulting vectors locally using Chroma (FAISS-like vector DB).
    """

    def __init__(self) -> None:
        """
        Initialize the embedder with:
        - A multilingual embedding model
        - MinIO loader instance for object storage interaction
        """
        print(f"-> Đang khởi tạo embeddings cho model: {MODEL_NAME}")
        self.embeddings = get_embeddings()
        self.minio_loader = MinioLoader(
            MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
        )

    def document_embedding_vectorstore(
        self,
        splits: list[Document],
        collection_name: str,
        persist_directory: str,
    ) -> Chroma:
        """
        Generate and store document embeddings into a local Chroma vector store.

        Args:
            splits (list[Document]): List of LangChain `Document` objects,
            each representing a text chunk.
            collection_name (str): Name of the vector store collection
            (acts as an identifier).
            persist_directory (str): Local filesystem path to persist the vector index.

        Returns:
            Chroma: The initialized Chroma vector store with the embedded documents.
        """
        print("========= Initializing Chroma Vector Store =============")

        # 1. Initialize Chroma vector store with persistence
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        # 2. Generate unique UUIDs for each document chunk
        uuids = [str(uuid4()) for _ in splits]

        # 3. Clean up metadata to avoid nested/unserializable fields
        print("Filtering complex metadata before storing...")
        filtered_splits = filter_complex_metadata(splits)

        # 4. Store the documents with corresponding UUIDs
        print(f"Adding {len(filtered_splits)} document chunks to vector store…")
        vectordb.add_documents(documents=filtered_splits, ids=uuids)

        return vectordb
