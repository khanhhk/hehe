from typing import List

import torch
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


class EmbeddingService(Embeddings):
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(model_name).to(device)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single input string as a normalized vector.

        Args:
            text (str): The input text to be embedded.

        Returns:
            List[float]: The normalized embedding vector as a list of floats.
        """
        vector = self.embedding_model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of input strings as normalized vectors.

        Args:
            texts (List[str]): A list of input texts to be embedded.

        Returns:
            List[List[float]]: A list of normalized embedding vectors.
        """
        vectors = self.embedding_model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
