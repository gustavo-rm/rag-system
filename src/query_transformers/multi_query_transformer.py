from typing import List
import logging
import re
from .base import QueryTransformer
from src.components.llm import LLM

# Configuração de Logger
logger = logging.getLogger(__name__)


class MultiQueryTransformer(QueryTransformer):
    """
    Transforma a consulta gerando múltiplas variações para aumentar a revocação (recall).
    Inclui proteções contra excesso de variações para manter a performance.
    Especializado em manter o idioma Português e expandir sinônimos.
    """

    def __init__(self, llm: LLM, num_queries: int = 3):
        self.llm = llm
        self.num_queries = num_queries
        # Prompt otimizado para instruir o modelo a não numerar, mas há uma regex caso ele numere.
        self.prompt_template = """
            Você é um assistente de IA especialista em buscas geográficas e factuais em PORTUGUÊS.
            Sua tarefa é gerar {num} variações da pergunta do usuário para encontrar a resposta em documentos técnicos.

            Regras OBRIGATÓRIAS:
            1. Responda APENAS em PORTUGUÊS DO BRASIL.
            2. Use sinônimos técnicos. (Ex: "maior montanha" -> "ponto culminante", "pico mais alto", "altitude máxima").
            3. NÃO responda à pergunta. Apenas reescreva as variações.
            4. NÃO escreva introduções como "Aqui estão as variações". Retorne APENAS as perguntas, uma por linha.

            Pergunta Original: "{question}"
            """

    def _sanitize_response(self, response_text: str) -> List[str]:
        """
        Método auxiliar para limpar a saída 'suja' do LLM.
        Remove numeração (1., 2.), bullets (-, *), aspas e linhas vazias.
        """
        cleaned = []
        lines = response_text.split('\n')

        for line in lines:
            line = line.strip()

            # Pula linhas vazias
            if not line:
                continue

            # Regex:
            # ^ : Começo da linha
            # [\d\-\*\•]+ : Qualquer combinação de dígitos, hífens, asteriscos ou bullets
            # [\.\)\s]* : Seguido opcionalmente de ponto, parêntese ou espaços
            # Ex: Remove "1.", "1 -", "- ", "* ", "2)"
            line = re.sub(r'^[\d\-\*\•]+[\.\)\s]*', '', line)

            # Remove aspas extras que modelos gostam de colocar
            line = line.strip('"\'')

            if len(line) > 5:  # Ignora linhas muito curtas/lixo
                cleaned.append(line)

        return cleaned

    def transform(self, query: str) -> List[str]:
        """
        Gera variações e garante que a query original seja a primeira.
        """
        logger.info(f"⚡ MultiQuery: Gerando variações para: '{query}'")

        try:
            response = self.llm.generate_response(
                prompt=self.prompt_template.format(num=self.num_queries, question=query),
                system_prompt="Gerador de variações de busca em Português.",
                max_new_tokens=150,
                temperature=0.5
            )

            # 1. Limpeza
            variations = self._sanitize_response(response)

            # 2. Guardrail de Quantidade
            # Pega apenas as 'num_queries' primeiras variações, ignorando o resto
            # para proteger a performance do Retriever.
            limited_variations = variations[:self.num_queries]

            if not limited_variations:
                logger.warning("MultiQuery não gerou variações válidas. Usando apenas original.")
                return [query]

            logger.info(f"Variações geradas: {limited_variations}")

            # Retorna [Original] + [Variações Limitadas]
            # A original sempre vai primeiro pois é a intenção real do usuário
            return [query] + limited_variations

        except Exception as e:
            logger.error(f"Erro no MultiQueryTransformer: {e}. Retornando query original.")
            return [query]