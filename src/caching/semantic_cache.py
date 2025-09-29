import faiss
import numpy as np
from typing import Optional, List


class SemanticCache:
    """
    Um cache em memória que armazena respostas para perguntas com base na
    similaridade semântica de seus embeddings.
    """

    def __init__(self, dimension: int, similarity_threshold: float = 0.98):
        """
        Inicializa o cache semântico.

        Args:
            dimension (int): A dimensão dos vetores de embedding (ex: 768).
            similarity_threshold (float): O limiar para considerar duas perguntas
                                          como "iguais" (um cache hit).
        """
        # FAISS usa L2 (distância Euclidiana). Para similaridade de cosseno,
        # normalizamos os vetores e usamos IndexFlatIP (Produto Interno).
        self.index = faiss.IndexFlatIP(dimension)
        self.responses: List[str] = []
        self.threshold = similarity_threshold
        print(f"Cache Semântico inicializado com dimensão {dimension} e limiar {similarity_threshold}.")

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normaliza os vetores para a busca de similaridade de cosseno."""
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, question_embedding: np.ndarray, answer: str):
        """
        Adiciona um novo par (embedding da pergunta, resposta) ao cache.
        """
        # FAISS espera um array 2D
        if question_embedding.ndim == 1:
            question_embedding = np.expand_dims(question_embedding, axis=0)

        normalized_embedding = self._normalize(question_embedding)
        self.index.add(normalized_embedding)
        self.responses.append(answer)

    def check(self, query_embedding: np.ndarray) -> Optional[str]:
        """
        Verifica se uma pergunta similar já existe no cache.

        Retorna:
            A resposta em cache se um hit for encontrado, caso contrário None.
        """
        if self.index.ntotal == 0:
            return None  # Cache está vazio

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        normalized_query = self._normalize(query_embedding)

        # Busca pelo vizinho mais próximo (k=1)
        distances, indices = self.index.search(normalized_query, 1)

        # distances aqui é na verdade a similaridade de cosseno (devido à normalização e IndexFlatIP)
        top_similarity = distances[0][0]
        top_index = indices[0][0]

        print(f"DEBUG: Similaridade encontrada no cache: {top_similarity:.4f} | Limiar: {self.threshold}")

        if top_similarity >= self.threshold:
            print(f"INFO: Cache hit! Similaridade de {top_similarity:.4f} (acima de {self.threshold}).")
            return self.responses[top_index]

        return None