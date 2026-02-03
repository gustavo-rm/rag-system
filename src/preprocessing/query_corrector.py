import re
import logging
import unicodedata
from typing import Optional
from src.config import Config

logger = logging.getLogger(__name__)


class QueryCorrector:
    """
    Responsible for:
    - Structural text cleaning
    - Optional grammar correction (LanguageTool)
    - Normalization for lexical search engines (BM25)

    Designed to act BEFORE:
    - exact cache
    - semantic cache (FAISS)
    - embedding generation
    """

    def __init__(
        self,
        language: str = Config.LANGUAGE,
        enable_grammar: bool = False
    ):
        """
        Initializes the QueryCorrector.

        Args:
            language (str): Language code (e.g., 'pt-BR' or 'en-US').
            enable_grammar (bool): Activates grammar correction (high cost).
        """
        self.language = language
        self.enable_grammar = enable_grammar
        self._tool: Optional[object] = None

        if self.enable_grammar:
            self._initialize_language_tool()

    # Initialization

    def _initialize_language_tool(self) -> None:
        """
        Initializes the LanguageTool library if available.
        Handles import errors and exceptions gracefully.
        """
        try:
            import language_tool_python
            self._tool = language_tool_python.LanguageTool(self.language)
            logger.info(
                f"QueryCorrector: LanguageTool activated ({self.language})."
            )
        except ImportError:
            logger.warning(
                "QueryCorrector: 'language_tool_python' not installed. "
                "Grammar correction disabled."
            )
            self._tool = None
        except Exception as e:
            logger.warning(
                f"QueryCorrector: Failed to start LanguageTool: {e}. "
                "Grammar correction disabled."
            )
            self._tool = None

    # Public API

    def correct_query(self, text: str) -> str:
        """
        Performs structural cleaning and, optionally, grammar correction.

        Ideal for:
        - text displayed to the user
        - LLM input

        Args:
            text (str): The input text to correct.

        Returns:
            str: The corrected text.
        """
        if not text:
            return ""

        text = self._normalize_whitespace(text)

        if self._tool:
            try:
                text = self._tool.correct(text)
            except Exception as e:
                logger.error(
                    f"QueryCorrector: error in grammar correction: {e}. "
                    "Text kept as is."
                )

        return text

    def normalize_for_bm25(self, text: str) -> str:
        """
        Normalizes text for lexical search (BM25 / inverted index).

        Strategy:
        - Removes accents
        - Removes punctuation
        - Converts to lowercase

        Args:
            text (str): The text to normalize.

        Returns:
            str: The normalized text.
        """
        if not text:
            return ""

        text = self._remove_accents(text)
        text = self._remove_punctuation(text)
        text = text.lower()

        return self._normalize_whitespace(text)

    # Internal Methods (Single Responsibility)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Removes line breaks and duplicate spaces."""
        return " ".join(text.split())

    @staticmethod
    def _remove_accents(text: str) -> str:
        """Removes Unicode accents (NFKD)."""
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(
            c for c in normalized if not unicodedata.combining(c)
        )

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """
        Removes punctuation, keeping letters, numbers, and spaces.
        Ideal for BM25.
        """
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)
