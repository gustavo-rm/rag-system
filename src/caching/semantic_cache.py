import faiss
import logging
import numpy as np
from typing import Optional, List

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Cache de Similaridade Semântica usando FAISS.

    Armazena vetores de perguntas passadas. Se uma nova pergunta for semanticamente
    próxima (acima de um limiar), retorna a resposta antiga, economizando chamadas de LLM.
    """

    def __init__(self, dimension: int, similarity_threshold: float = 0.92):
        """
        Inicializa o índice vetorial.

        Args:
            dimension (int): Dimensão dos embeddings (ex: 768 para MPNet, 1536 para OpenAI).
            similarity_threshold (float): Nota de corte (0.0 a 1.0) para considerar similaridade.
                                          Recomenda-se 0.90+ para evitar respostas erradas.
        """
        self.dimension = dimension
        self.threshold = similarity_threshold

        # IndexFlatIP = Inner Product (Produto Interno).
        # Com vetores normalizados, isso equivale à Similaridade de Cosseno.
        self.index = faiss.IndexFlatIP(dimension)

        # Armazena as respostas textuais alinhadas com os índices do FAISS
        self.responses: List[str] = []

        logger.info(f"🧠 Cache Semântico (Camada 2) inicializado. Dim: {dimension}, Threshold: {similarity_threshold}")

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Prepara o vetor para o FAISS: garante float32, formato 2D e normalização.

        IMPORTANTE: Cria uma cópia para não alterar o vetor original in-place.
        """
        # Garante que é float32 (FAISS exige isso)
        vec = vector.astype(np.float32)

        # Garante formato 2D (1, dim)
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)

        # Normalização L2 para usar Cosseno
        faiss.normalize_L2(vec)
        return vec

    def add(self, question_embedding: np.ndarray, answer: str):
        """
        Adiciona uma nova entrada ao cache.
        """
        # Copia e prepara o vetor
        normalized_embedding = self._prepare_vector(question_embedding)

        self.index.add(normalized_embedding)
        self.responses.append(answer)
        logger.debug("Nova entrada adicionada ao Cache Semântico.")

    def check(self, query_embedding: np.ndarray) -> Optional[str]:
        """
        Verifica se existe alguma pergunta similar no histórico.

        Args:
            query_embedding (np.ndarray): O vetor da nova pergunta.

        Returns:
            Optional[str]: A resposta cacheada se houver similaridade suficiente.
        """
        if self.index.ntotal == 0:
            return None

        normalized_query = self._prepare_vector(query_embedding)

        # Busca o 1 vizinho mais próximo (k=1)
        # D: Distâncias (Scores), I: Índices
        D, I = self.index.search(normalized_query, 1)

        top_score = D[0][0]
        top_index = I[0][0]

        logger.debug(f"Cache Semântico: Score encontrado {top_score:.4f} (Limiar: {self.threshold})")

        if top_score >= self.threshold:
            logger.info(f"🎯 Cache Semântico HIT! Score: {top_score:.4f}")
            if 0 <= top_index < len(self.responses):
                return self.responses[top_index]
            else:
                logger.error("Índice do FAISS dessincronizado com lista de respostas.")
                return None

        return None