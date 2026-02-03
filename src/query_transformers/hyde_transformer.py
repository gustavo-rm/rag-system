from typing import List
from .base import QueryTransformer
from src.components.llm import LLM
from src.prompts import Prompts
import logging

# Logger Configuration
logger = logging.getLogger(__name__)


class HyDETransformer(QueryTransformer):
    """
    Transforms the query by generating a Hypothetical Document (HyDE).

    HyDE (Hypothetical Document Embeddings) improves retrieval by generating
    a fake but semantically relevant document answer, which is then used
    for similarity search instead of the raw question.
    """

    def __init__(self, llm: LLM):
        """
        Initializes the HyDE transformer.

        Args:
            llm (LLM): The Language Model used to generate the hypothetical document.
        """
        self.llm = llm
        self.prompt_template = Prompts.HYDE_PROMPT_TEMPLATE

    def transform(self, query: str) -> List[str]:
        """
        Generates a hypothetical document and returns it along with the original query.

        Args:
            query (str): The original user query.

        Returns:
            List[str]: A list containing the original query and the hypothetical document.
        """
        logger.info(f"⚡ HyDE: Generating hypothetical document for: '{query}'")

        hypothetical_doc = self.llm.generate_response(
            prompt=self.prompt_template.format(question=query),
            system_prompt=Prompts.HYDE_SYSTEM_PROMPT,
            temperature=0.4,
            max_new_tokens=120
        )

        # Returns the Original Query AND the Hypothetical Document.
        # The HybridRetriever will search for both. The query catches exact keywords (BM25).
        # HyDE catches semantic similarity (Vector).
        return [query, hypothetical_doc]
