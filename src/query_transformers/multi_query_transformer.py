from typing import List
from .base import QueryTransformer
from src.components.llm import LLM


class MultiQueryTransformer(QueryTransformer):
    """Transforma a consulta gerando múltiplas variações da mesma pergunta."""

    def __init__(self, llm: LLM, num_queries: int = 3):
        self.llm = llm
        self.num_queries = num_queries
        self.prompt_template = """
        Gere {num} variações da pergunta abaixo para melhorar a busca em documentos.
        Dê foco em sinônimos e termos técnicos alternativos.
        Separe por quebras de linha. Não numere.
        Pergunta: {question}
        """

    def transform(self, query: str) -> List[str]:
        print(f"⚡ MultiQuery: Gerando {self.num_queries} variações...")

        response = self.llm.generate_response(
            prompt=self.prompt_template.format(num=self.num_queries, question=query),
            system_prompt="Assistente de reescrita de query.",
            temperature=0.7  # Mais criatividade aqui ajuda
        )

        # Limpeza robusta: remove números (1.), bullets (-) e linhas vazias
        lines = response.split('\n')
        cleaned_queries = []
        for line in lines:
            line = line.strip()
            # Remove "1. ", "2. ", "- " do início
            if len(line) > 0:
                if line[0] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
                    line = line.lstrip('0123456789.- ')
                cleaned_queries.append(line)

        # Garante que a original está na lista
        return [query] + cleaned_queries