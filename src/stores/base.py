from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStore(ABC):
    @abstractmethod
    def store_embeddings(self,
                         chunks: List[str],
                         embeddings: List[List[float]],
                         ids: List[str] = None,
                         metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Armazena textos, vetores e metadados.

        Args:
            chunks: Lista de textos.
            embeddings: Lista de vetores.
            ids: Lista de IDs únicos (opcional).
            metadatas: Lista de dicionários com dados extras (fonte, página, etc).
        """
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete(self):
        pass