import re
import logging
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)


class QueryCorrector:
    """
    Responsável por:
    - Limpeza estrutural de texto
    - Correção gramatical opcional (LanguageTool)
    - Normalização para mecanismos de busca léxica (BM25)

    Projetado para atuar ANTES de:
    - cache exato
    - cache semântico (FAISS)
    - geração de embeddings
    """

    def __init__(
        self,
        language: str = "pt-BR",
        enable_grammar: bool = False
    ):
        """
        Args:
            language (str): Código do idioma (ex: 'pt-BR').
            enable_grammar (bool): Ativa correção gramatical (custo alto).
        """
        self.language = language
        self.enable_grammar = enable_grammar
        self._tool: Optional[object] = None

        if self.enable_grammar:
            self._initialize_language_tool()

    # Inicialização

    def _initialize_language_tool(self) -> None:
        try:
            import language_tool_python
            self._tool = language_tool_python.LanguageTool(self.language)
            logger.info(
                f"QueryCorrector: LanguageTool ativado ({self.language})."
            )
        except ImportError:
            logger.warning(
                "QueryCorrector: 'language_tool_python' não instalado. "
                "Correção gramatical desativada."
            )
            self._tool = None
        except Exception as e:
            logger.warning(
                f"QueryCorrector: Falha ao iniciar LanguageTool: {e}. "
                "Correção gramatical desativada."
            )
            self._tool = None

    # API Pública

    def correct_query(self, text: str) -> str:
        """
        Executa limpeza estrutural e, opcionalmente, correção gramatical.

        Ideal para:
        - texto exibido ao usuário
        - entrada de LLM
        """
        if not text:
            return ""

        text = self._normalize_whitespace(text)

        if self._tool:
            try:
                text = self._tool.correct(text)
            except Exception as e:
                logger.error(
                    f"QueryCorrector: erro na correção gramatical: {e}. "
                    "Texto mantido."
                )

        return text

    def normalize_for_bm25(self, text: str) -> str:
        """
        Normaliza texto para busca léxica (BM25 / inverted index).

        Estratégia:
        - Remove acentos
        - Remove pontuação
        - Converte para lowercase
        """
        if not text:
            return ""

        text = self._remove_accents(text)
        text = self._remove_punctuation(text)
        text = text.lower()

        return self._normalize_whitespace(text)

    # Métodos Internos (Responsabilidade Única)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Remove quebras de linha e espaços duplicados."""
        return " ".join(text.split())

    @staticmethod
    def _remove_accents(text: str) -> str:
        """Remove acentuação Unicode (NFKD)."""
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(
            c for c in normalized if not unicodedata.combining(c)
        )

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """
        Remove pontuação mantendo letras, números e espaços.
        Ideal para BM25.
        """
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)
