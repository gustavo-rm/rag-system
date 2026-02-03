from typing import List
import logging
import re
from .base import QueryTransformer
from src.components.llm import LLM
from src.prompts import Prompts

# Logger Configuration
logger = logging.getLogger(__name__)


class MultiQueryTransformer(QueryTransformer):
    """
    Transforms the query by generating multiple variations to increase recall.
    Includes safeguards against excessive variations to maintain performance.
    Specialized in maintaining Portuguese language and expanding synonyms.
    """

    def __init__(self, llm: LLM, num_queries: int = 3):
        """
        Initializes the MultiQueryTransformer.

        Args:
            llm (LLM): The Language Model used to generate query variations.
            num_queries (int): The number of variations to generate.
        """
        self.llm = llm
        self.num_queries = num_queries
        # Optimized prompt to instruct the model not to number, but there is a regex in case it does.
        self.prompt_template = Prompts.MULTI_QUERY_PROMPT_TEMPLATE

    def _sanitize_response(self, response_text: str) -> List[str]:
        """
        Helper method to clean 'dirty' LLM output.
        Removes numbering (1., 2.), bullets (-, *), quotes, and empty lines.

        Args:
            response_text (str): The raw text response from the LLM.

        Returns:
            List[str]: A list of cleaned query strings.
        """
        cleaned = []
        lines = response_text.split('\n')

        for line in lines:
            line = line.strip()

            # Skips empty lines
            if not line:
                continue

            # Regex:
            # ^ : Start of line
            # [\d\-\*\•]+ : Any combination of digits, hyphens, asterisks, or bullets
            # [\.\)\s]* : Optionally followed by dot, parenthesis, or spaces
            # Ex: Removes "1.", "1 -", "- ", "* ", "2)"
            line = re.sub(r'^[\d\-\*\•]+[\.\)\s]*', '', line)

            # Removes extra quotes that models like to add
            line = line.strip('"\'')

            if len(line) > 5:  # Ignores very short lines/garbage
                cleaned.append(line)

        return cleaned

    def transform(self, query: str) -> List[str]:
        """
        Generates variations and ensures the original query is the first one.

        Args:
            query (str): The original user query.

        Returns:
            List[str]: A list of queries starting with the original, followed by variations.
        """
        logger.info(f"⚡ MultiQuery: Generating variations for: '{query}'")

        try:
            response = self.llm.generate_response(
                prompt=self.prompt_template.format(num=self.num_queries, question=query),
                system_prompt=Prompts.MULTI_QUERY_SYSTEM_PROMPT,
                max_new_tokens=150,
                temperature=0.5
            )

            # 1. Cleaning
            variations = self._sanitize_response(response)

            # 2. Quantity Guardrail
            # Takes only the first 'num_queries' variations, ignoring the rest
            # to protect Retriever performance.
            limited_variations = variations[:self.num_queries]

            if not limited_variations:
                logger.warning("MultiQuery did not generate valid variations. Using only original.")
                return [query]

            logger.info(f"Generated variations: {limited_variations}")

            # Returns [Original] + [Limited Variations]
            # The original always goes first as it is the user's real intent
            return [query] + limited_variations

        except Exception as e:
            logger.error(f"Error in MultiQueryTransformer: {e}. Returning original query.")
            return [query]
