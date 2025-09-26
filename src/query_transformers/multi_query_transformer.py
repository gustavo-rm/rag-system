from typing import List
from .base import QueryTransformer
from src.components.llm import LLM


class MultiQueryTransformer(QueryTransformer):
    """Transforma a consulta gerando múltiplas variações da mesma pergunta."""

    def __init__(self, llm: LLM, num_queries: int = 3):
        self.llm = llm
        self.num_queries = num_queries
        self.mq_prompt_template = """
        Você é um assistente de IA especialista em reescrever perguntas de usuários para otimizar a busca vetorial.
        Gere {num_queries} versões alternativas e semanticamente distintas da seguinte pergunta.
        Cada pergunta deve explorar uma faceta diferente ou perspectiva da pergunta original.
        Retorne APENAS as perguntas, separadas por uma quebra de linha. Não adicione números ou marcadores.

        Pergunta Original: {question}

        Perguntas Alternativas:
        """
        self.mq_system_prompt = "Você é um assistente de reescrita de perguntas."

    def transform(self, query: str) -> List[str]:
        print(f"Usando estratégia: Multi-Query (Gerando {self.num_queries} variações).")
        prompt = self.mq_prompt_template.format(question=query, num_queries=self.num_queries)
        generated_queries_str = self.llm.generate_response(
            prompt=prompt,
            system_prompt=self.mq_system_prompt
        )

        generated_queries = [q.strip() for q in generated_queries_str.split('\n') if q.strip()]
        all_queries = [query] + generated_queries

        print("  - Perguntas Geradas:")
        for q in all_queries:
            print(f"    - {q}")

        return all_queries