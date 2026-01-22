import logging
import os
import pandas as pd
from typing import List, Dict, Optional, Any
from datasets import Dataset

# Ragas Metrics
from ragas import evaluate
from ragas.metrics import (
    faithfulness,  # Did the model hallucinate? (Generation)
    answer_relevancy,  # Did it answer what was asked? (Generation)
    context_precision,  # Did the Retriever bring useful documents at the top? (Retrieval)
    context_recall,  # Did the Retriever bring ALL necessary information? (Retrieval)
)
# LangChain Integration (Required for RAGAS to work well)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Logger Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Semantic Evaluator for RAG (Retrieval-Augmented Generation) systems.

    Uses LLM-based metrics, which assess the meaning and truthfulness of responses.

    This class acts as an independent 'Judge'. It is recommended to use a strong
    model (e.g., GPT-4o) for evaluation, independent of the model used in the RAG.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initializes the evaluator by configuring the 'Judge' LLM.

        Args:
            openai_api_key (str): OpenAI Key. If None, attempts to retrieve from the environment.
                                  RAGAS works best with OpenAI as a judge.
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning(
                "⚠️ RAGEvaluator: OpenAI API Key not found. Evaluation may fail if no global configuration exists.")

        # Configure LLM and Embeddings specifically for RAGAS (Judge)
        # Using gpt-4o-mini or gpt-4 for evaluation as they are more rigorous
        self.judge_llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.api_key)
        self.judge_embeddings = OpenAIEmbeddings(api_key=self.api_key)

        # Metrics divided by use case
        self.metrics_with_ground_truth = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

        self.metrics_no_ground_truth = [
            faithfulness,
            answer_relevancy
        ]

        logger.info("⚖️ RAGEvaluator (v3.1) initialized with OpenAI Judge.")

    def evaluate_single(self,
                        question: str,
                        generated_answer: str,
                        retrieved_contexts: List[str],
                        ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluates a single RAG interaction.

        Args:
            question (str): The user's question.
            generated_answer (str): The Chatbot's final response.
            retrieved_contexts (List[str]): List of retrieved texts (context).
            ground_truth (str, optional): The ideal answer (answer key).

        Returns:
            Dict[str, float]: Dictionary with scores (0.0 to 1.0).
        """
        # Prepares data in the format RAGAS expects
        data = {
            "question": [question],
            "answer": [generated_answer],
            "contexts": [retrieved_contexts],
        }

        metrics_to_use = self.metrics_no_ground_truth

        if ground_truth:
            data["ground_truth"] = [ground_truth]
            metrics_to_use = self.metrics_with_ground_truth
            logger.info("Running full evaluation (with Ground Truth)...")
        else:
            logger.info("Running partial evaluation (no Ground Truth - Generation Only)...")

        try:
            dataset = Dataset.from_dict(data)

            # Executes evaluation
            # Explicitly passing llm and embeddings to ensure RAGAS uses our config
            results = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
                llm=self.judge_llm,
                embeddings=self.judge_embeddings,
                raise_exceptions=False
            )

            # Convert to standard Python dictionary
            # The RAGAS results object behaves like a dict
            scores = {k: round(v, 4) for k, v in results.items()}

            logger.info(f"📊 Evaluation Results: {scores}")
            return scores

        except Exception as e:
            logger.error(f"❌ Failed to run RAGAS: {e}")
            return {"error": 0.0}

    def evaluate_batch(self, samples: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Evaluates a batch of questions (Golden Dataset) and returns a DataFrame.

        Args:
            samples (List[Dict]): List of dicts containing keys:
                                  'question', 'answer', 'contexts', 'ground_truth' (optional).

        Returns:
            pd.DataFrame: Table with comparative results.
        """
        logger.info(f"Starting batch evaluation of {len(samples)} items...")

        # Transforms list of dicts into dict of lists (Dataset columnar format)
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        has_gt = all("ground_truth" in s for s in samples)

        for s in samples:
            data["question"].append(s["question"])
            data["answer"].append(s["answer"])
            data["contexts"].append(s["contexts"])
            if has_gt:
                data["ground_truth"].append(s.get("ground_truth", ""))

        if not has_gt:
            del data["ground_truth"]
            metrics = self.metrics_no_ground_truth
        else:
            metrics = self.metrics_with_ground_truth

        dataset = Dataset.from_dict(data)

        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.judge_llm,
            embeddings=self.judge_embeddings
        )

        df = results.to_pandas()
        logger.info("Batch evaluation completed.")
        return df
