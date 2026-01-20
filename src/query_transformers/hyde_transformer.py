from typing import List
from .base import QueryTransformer
from src.components.llm import LLM
import logging
# Configuração de Logger
logger = logging.getLogger(__name__)


class HyDETransformer(QueryTransformer):
    """Transforma a consulta gerando um documento hipotético (HyDE)."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.prompt_template = """
        Escreva um breve trecho técnico que responda à pergunta abaixo. 
        Não responda a pergunta diretamente, mas simule como seria o texto em um manual técnico que contém a resposta.
        Pergunta: {question}
        Passagem do manual:
        """

    def transform(self, query: str) -> List[str]:
        logger.info(f"⚡ HyDE: Gerando documento hipotético para: '{query}'")

        hypothetical_doc = self.llm.generate_response(
            prompt=self.prompt_template.format(question=query),
            system_prompt="Você é um gerador de dados sintéticos para RAG.",
            temperature=0.4,
            max_new_tokens=120
        )

        # Retorna a Query Original E o Documento Hipotético.
        # O HybridRetriever vai buscar ambos. A query pega palavras-chave exatas (BM25).
        # O HyDE pega a similaridade semântica (Vetor).
        return [query, hypothetical_doc]