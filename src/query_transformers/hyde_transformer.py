from typing import List
from .base import QueryTransformer
from src.components.llm import LLM

class HyDETransformer(QueryTransformer):
    """Transforma a consulta gerando um documento hipotético (HyDE)."""
    def __init__(self, llm: LLM):
        self.llm = llm
        self.hyde_prompt_template = """
        Gere um parágrafo curto e conciso que responda à seguinte pergunta.
        Responda como se você fosse um trecho de um documento de referência, focando nos fatos.
        Pergunta: {question}
        Parágrafo de resposta hipotética:
        """
        self.hyde_system_prompt = "Você é um assistente que gera documentos factuais hipotéticos."

    def transform(self, query: str) -> List[str]:
        print("Usando estratégia: HyDE (Documento Hipotético).")
        prompt = self.hyde_prompt_template.format(question=query)
        hypothetical_document = self.llm.generate_response(
            prompt=prompt,
            system_prompt=self.hyde_system_prompt,
            temperature=0.3
        )
        print(f"  - Documento Hipotético Gerado: {hypothetical_document[:100]}...")
        return [hypothetical_document]