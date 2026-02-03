import torch
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List
import logging
from src.config import Config

# Logger Configuration
logger = logging.getLogger(__name__)


class ReRanker:
    """
    Responsible for search refinement (Stage 2).

    Receives candidate texts and uses a Cross-Encoder to calculate
    the probability of real relevance in relation to the question.
    """

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initializes the Re-ranking model.

        Args:
            model_name (str): Name of the Cross-Encoder model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name or Config.DEFAULT_RERANKER_MODEL

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"🔄 Initializing ReRanker ({self.model_name}) on: {self.device}...")

        try:
            if self.device == "cuda":
                # Optimized loading for GPU (FP16)
                self.model = CrossEncoder(self.model_name, device="cpu", automodel_args={"torch_dtype": torch.float16})
                self.model.model.to("cuda")
                logger.info("🚀 ReRanker loaded on CUDA (FP16).")
            else:
                self.model = CrossEncoder(self.model_name, device="cpu")
                logger.info("✅ ReRanker loaded on CPU.")

        except Exception as e:
            logger.warning(f"⚠️ Error loading on GPU ({e}). Using CPU.")
            self.model = CrossEncoder(self.model_name, device="cpu")
            self.device = "cpu"

    def rerank(self, query: str, documents: List[str], top_n: int = 3, threshold: float = 0.01) -> List[str]:
        """
        Reranks a list of texts.

        Args:
            query (str): The user's question.
            documents (List[str]): List of raw TEXT strings coming from the Retriever.
            top_n (int): Number of documents to keep.
            threshold (float): Minimum score (0 to 1) to consider relevant.
                               Lowered to 0.01 to avoid false negatives in BGE.

        Returns:
            List[str]: List of the best texts ordered by relevance.
        """
        if not documents:
            logger.warning("ReRanker received empty list.")
            return []

        # Remove exact duplicates and empty strings before processing
        unique_docs = list(set([doc for doc in documents if doc and doc.strip()]))

        if not unique_docs:
            return []

        # Prepare pairs [Question, Text]
        pairs = [[query, doc] for doc in unique_docs]

        # Prediction (Raw Logits: can be negative, e.g., -8.5 to +2.1)
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Ensure numpy array even if it's 1 document
        if isinstance(scores, float):
            scores = np.array([scores])
        elif isinstance(scores, list):
            scores = np.array(scores)

        # Sigmoid Normalization (Transforms Logits into Probability 0.0 to 1.0)
        # Ex: Logit -2 becomes 0.12. Logit 5 becomes 0.99.
        scores_sig = 1 / (1 + np.exp(-scores))

        # Combines (Text, Score)
        scored_results = []
        for doc, score in zip(unique_docs, scores_sig):
            # Debug log to adjust threshold if needed
            # logger.debug(f"Score: {score:.4f} | Text: {doc[:30]}...")

            if score >= threshold:
                scored_results.append((doc, score))

        # If threshold filtered everything, but we have documents, take the "least worst" (Fallback)
        if not scored_results and unique_docs:
            logger.warning("ReRanker: Threshold too high. Returning Top-1 'least worst' as fallback.")
            best_idx = np.argmax(scores_sig)
            return [unique_docs[best_idx]]

        # Sort descending by score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Return only the texts
        final_docs = [doc for doc, score in scored_results[:top_n]]

        return final_docs
