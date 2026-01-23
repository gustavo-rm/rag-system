import faiss
import logging
import numpy as np
from typing import Optional, List

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic Similarity Cache using FAISS.

    Stores vectors of past questions. If a new question is semantically
    close (above a threshold), it returns the old response, saving LLM calls.
    """

    def __init__(self, dimension: int, similarity_threshold: float = 0.92):
        """
        Initializes the vector index.

        Args:
            dimension (int): Dimension of embeddings (e.g., 768 for MPNet, 1536 for OpenAI).
            similarity_threshold (float): Cutoff score (0.0 to 1.0) to consider similarity.
                                          Recommended 0.90+ to avoid incorrect answers.
        """
        self.dimension = dimension
        self.threshold = similarity_threshold

        # IndexFlatIP = Inner Product.
        # With normalized vectors, this is equivalent to Cosine Similarity.
        self.index = faiss.IndexFlatIP(dimension)

        # Stores textual responses aligned with FAISS indices
        self.responses: List[str] = []

        logger.info(f"🧠 Semantic Cache (Layer 2) initialized. Dim: {dimension}, Threshold: {similarity_threshold}")

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Prepares the vector for FAISS: ensures float32, 2D format, and normalization.

        IMPORTANT: Creates a copy to not alter the original vector in-place.

        Args:
            vector (np.ndarray): The input vector.

        Returns:
            np.ndarray: The prepared vector.
        """
        # Ensures it is float32 (FAISS requires this)
        vec = vector.astype(np.float32)

        # Ensures 2D format (1, dim)
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)

        # L2 Normalization to use Cosine
        faiss.normalize_L2(vec)
        return vec

    def add(self, question_embedding: np.ndarray, answer: str):
        """
        Adds a new entry to the cache.

        Args:
            question_embedding (np.ndarray): The embedding of the question.
            answer (str): The answer to cache.
        """
        # Copies and prepares the vector
        normalized_embedding = self._prepare_vector(question_embedding)

        self.index.add(normalized_embedding)
        self.responses.append(answer)
        logger.debug("New entry added to Semantic Cache.")

    def check(self, query_embedding: np.ndarray) -> Optional[str]:
        """
        Checks if there is any similar question in the history.

        Args:
            query_embedding (np.ndarray): The vector of the new question.

        Returns:
            Optional[str]: The cached response if there is sufficient similarity, otherwise None.
        """
        if self.index.ntotal == 0:
            return None

        normalized_query = self._prepare_vector(query_embedding)

        # Searches for the 1 nearest neighbor (k=1)
        # D: Distances (Scores), I: Indices
        D, I = self.index.search(normalized_query, 1)

        top_score = D[0][0]
        top_index = I[0][0]

        logger.debug(f"Semantic Cache: Score found {top_score:.4f} (Threshold: {self.threshold})")

        if top_score >= self.threshold:
            logger.info(f"🎯 Semantic Cache HIT! Score: {top_score:.4f}")
            if 0 <= top_index < len(self.responses):
                return self.responses[top_index]
            else:
                logger.error("FAISS index out of sync with response list.")
                return None

        return None
