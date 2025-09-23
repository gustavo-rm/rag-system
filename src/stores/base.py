from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    @abstractmethod
    def store_embeddings(self, chunks: List[str], embeddings: List[List[float]], ids: List[str] = None):
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete(self):
        pass
