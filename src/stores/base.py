from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStore(ABC):
    """
    Abstract Base Class for vector storage operations.
    Defines the interface for storing, searching, and deleting embeddings.
    """

    @abstractmethod
    def store_embeddings(self,
                         chunks: List[str],
                         embeddings: List[List[float]],
                         ids: List[str] = None,
                         metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Stores texts, vectors, and metadata.

        Args:
            chunks (List[str]): List of texts to store.
            embeddings (List[List[float]]): List of embedding vectors.
            ids (List[str], optional): List of unique IDs.
            metadatas (List[Dict[str, Any]], optional): List of dictionaries with extra data (source, page, etc.).
        """
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for the most relevant vectors.

        Args:
            query_embedding (List[float]): The query vector.
            top_k (int): Number of results to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing match results (id, score, metadata).
        """
        pass

    @abstractmethod
    def delete(self):
        """
        Deletes the entire collection or index.
        """
        pass
