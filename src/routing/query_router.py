import logging
import json
from typing import Dict, Optional
from src.components.llm import LLM
from src.query_transformers import QueryTransformer
from src.prompts import Prompts

# Logger Configuration for this module
logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Semantic Query Router.

    This class analyzes the user's question intent and dynamically decides
    which RAG (Retrieval-Augmented Generation) strategy should be activated.

    Decision Mechanism:
    1. **Fast Heuristics:** Checks simple characteristics (length, patterns) for immediate decision.
    2. **Semantic Analysis (LLM):** Sends the question to the LLM with Few-Shot instructions for classification.

    Attributes:
        llm (LLM): Instance of the language model used for classification.
        strategies (Dict[str, QueryTransformer]): Mapping of strategy names ('noop', 'hyde') to instances.
    """

    def __init__(self, llm: LLM, strategies: Dict[str, QueryTransformer]):
        """
        Initializes the Router.

        Args:
            llm (LLM): The brain that will decide the route in complex cases.
            strategies (Dict[str, QueryTransformer]): The available tools.
        """
        self.llm = llm
        self.strategies = strategies

        # Detailed criteria to guide the LLM
        self.criteria = {
            "noop": Prompts.ROUTER_CRITERIA_NOOP,
            "hyde": Prompts.ROUTER_CRITERIA_HYDE,
            "multi_query": Prompts.ROUTER_CRITERIA_MULTI_QUERY
        }

        logger.info(f"📍 QueryRouter initialized with strategies: {list(self.strategies.keys())}")

    def route(self, question: str) -> QueryTransformer:
        """
        Selects the most appropriate transformation strategy for the question.

        Acts as an orchestrator (facade), delegating the decision to heuristics
        or the LLM, and returning the corresponding transformer object.

        Args:
            question (str): The original user question.

        Returns:
            QueryTransformer: The chosen strategy instance. Returns 'noop' in case of failure.
        """
        # 1. Attempt fast decision (Heuristics)
        heuristic_choice = self._check_heuristics(question)
        if heuristic_choice:
            logger.info(f"⚡ Router (Heuristics): Fast decision taken -> '{heuristic_choice.upper()}'")
            return self.strategies.get(heuristic_choice, self.strategies.get("noop"))

        # 2. Attempt deep decision (LLM)
        llm_choice = self._decide_via_llm(question)

        logger.info(f"✅ Router (LLM): Final decision -> '{llm_choice.upper()}' for '{question[:30]}...'")
        return self.strategies.get(llm_choice, self.strategies.get("noop"))

    def _check_heuristics(self, question: str) -> Optional[str]:
        """
        Applies deterministic rules based on the string structure.

        Args:
            question (str): The user's question.

        Returns:
            Optional[str]: The strategy name ('multi_query', etc) or None if no rule applies.
        """
        # Rule 1: Very short questions (< 4 words) usually need context expansion.
        # Ex: "Capital of Brazil" -> Needs to become "What is the capital..."
        if len(question.split()) < 4:
            return "multi_query"

        return None

    def _decide_via_llm(self, question: str) -> str:
        """
        Queries the LLM to classify the question's intent.

        Args:
            question (str): The user's question.

        Returns:
            str: The chosen strategy key (e.g., 'hyde'). Returns 'noop' in case of error.
        """
        logger.debug("🤔 Router (LLM): Analyzing question intent...")
        prompt = self._build_prompt(question)

        try:
            # Request a short and deterministic response
            response = self.llm.generate_response(
                prompt=prompt,
                system_prompt=Prompts.ROUTER_SYSTEM_PROMPT,
                max_new_tokens=10,
                temperature=0.0
            )

            # Delegates string cleaning and validation
            return self._parse_strategy_key(response)

        except Exception as e:
            logger.error(f"❌ Generic error in Router (LLM): {e}. Using fallback 'noop'.")
            return "noop"

    def _parse_strategy_key(self, raw_response: str) -> str:
        """
        Cleans the raw LLM response and maps it to a valid key.

        Args:
            raw_response (str): The text returned by the LLM (e.g., " 'multi_query' ", "hyde.", etc).

        Returns:
            str: A valid key present in `self.strategies` or 'noop' (fallback).
        """
        # Basic normalization
        cleaned = raw_response.strip().lower()
        cleaned = cleaned.replace("'", "").replace('"', "").replace(".", "")

        # 1. Exact Match
        if cleaned in self.strategies:
            return cleaned

        # 2. Fuzzy Match - In case the LLM is verbose
        if "multi" in cleaned or "query" in cleaned:
            return "multi_query"
        elif "hyde" in cleaned:
            return "hyde"
        elif "noop" in cleaned or "exact" in cleaned:
            return "noop"

        logger.warning(f"⚠️ Router returned unknown key: '{raw_response}'. Using fallback 'noop'.")
        return "noop"

    def _build_prompt(self, question: str) -> str:
        """
        Builds the classification prompt using Few-Shot Learning (Examples).

        Args:
            question (str): The question to be analyzed.

        Returns:
            str: The formatted prompt.
        """
        return Prompts.ROUTER_PROMPT_TEMPLATE.format(
            criteria_noop=self.criteria['noop'],
            criteria_hyde=self.criteria['hyde'],
            criteria_multi_query=self.criteria['multi_query'],
            question=question
        )
