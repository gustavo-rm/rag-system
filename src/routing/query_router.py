import logging
import json
from typing import Dict
from src.components.llm import LLM
from src.query_transformers import QueryTransformer

# Configuração do Logger para este módulo
logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Roteador Semântico de Consultas (Semantic Query Router).

    Esta classe analisa a intenção da pergunta do usuário e decide dinamicamente
    qual estratégia de RAG (Retrieval-Augmented Generation) deve ser ativada.

    Mecanismo de Decisão:
    1. **Heurística Rápida:** Verifica características simples (ex: tamanho da string) para
       decisões imediatas sem custo de LLM.
    2. **Análise Semântica (LLM):** Se a heurística não aplicar, envia a pergunta para o LLM
       com instruções para classificar a intenção e retornar um JSON estruturado.

    Attributes:
        llm (LLM): Instância do modelo de linguagem usado para a classificação.
        strategies (Dict[str, QueryTransformer]): Dicionário mapeando nomes (chaves)
                                                  para instâncias de transformadores (valores).
    """

    def __init__(self, llm: LLM, strategies: Dict[str, QueryTransformer]):
        """
        Inicializa o Roteador.

        Args:
            llm (LLM): O cérebro que decidirá a rota em casos complexos.
            strategies (Dict[str, QueryTransformer]): As ferramentas disponíveis.
                Exemplo:
                {
                    'noop': NoOpTransformer(),
                    'hyde': HyDETransformer(llm),
                    'multi_query': MultiQueryTransformer(llm)
                }
        """
        self.llm = llm
        self.strategies = strategies

        # Definições usadas no prompt para ajudar o LLM a decidir
        self.criteria = {
            "noop": "Perguntas muito específicas, técnicas, com códigos de erro, IDs, logs ou citações exatas.",
            "hyde": "Perguntas conceituais, pedidos de definição ('O que é...'), ou explicações teóricas onde contexto adicional ajuda.",
            "multi_query": "Perguntas curtas, vagas, ambíguas ou que podem ser descritas de várias formas diferentes."
        }

        logger.info(f"📍 QueryRouter inicializado com estratégias: {list(self.strategies.keys())}")

    def route(self, question: str) -> QueryTransformer:
        """
        Seleciona a estratégia de transformação mais adequada para a pergunta.

        Args:
            question (str): A pergunta original do usuário.

        Returns:
            QueryTransformer: A instância da estratégia escolhida (ex: objeto HyDETransformer).
                              Retorna a estratégia 'noop' (padrão) em caso de falha ou dúvida.
        """
        # 1. Heurística: Perguntas muito curtas geralmente precisam de expansão (MultiQuery)
        #    Economiza uma chamada de LLM.
        if len(question.split()) < 4:
            logger.info("⚡ Roteador (Heurística): Pergunta curta detectada. Roteando para 'multi_query'.")
            return self.strategies.get("multi_query", self.strategies.get("noop"))

        # 2. Decisão via LLM
        logger.debug("🤔 Roteador (LLM): Analisando intenção da pergunta...")
        prompt = self._build_prompt(question)

        try:
            # Solicita decisão em JSON para facilitar o parse
            response = self.llm.generate_response(
                prompt=prompt,
                system_prompt="Você é um classificador de intenção de busca (RAG Router). Responda apenas JSON.",
                max_new_tokens=15,  # Precisa de poucos tokens
                temperature=0.0  # Determinístico
            ).strip()

            # Limpeza defensiva de Markdown (caso o modelo responda ```json ... ```)
            clean_response = response.replace("```json", "").replace("```", "").strip()

            decision_data = json.loads(clean_response)
            chosen_key = decision_data.get("strategy", "noop").lower()

            if chosen_key in self.strategies:
                logger.info(f"✅ Roteador escolheu: '{chosen_key.upper()}' para a pergunta: '{question[:30]}...'")
                return self.strategies[chosen_key]
            else:
                logger.warning(f"⚠️ Roteador sugeriu chave desconhecida '{chosen_key}'. Usando fallback 'noop'.")

        except json.JSONDecodeError:
            logger.error(f"❌ Erro ao decodificar JSON do Roteador. Resposta bruta: '{response}'. Usando fallback.")
        except Exception as e:
            logger.error(f"❌ Erro genérico no Roteador: {e}. Usando fallback.")

        # Fallback seguro
        return self.strategies.get("noop")

    def _build_prompt(self, question: str) -> str:
        """Constrói o prompt de classificação."""
        return f"""
        Analise a pergunta e escolha a melhor estratégia de recuperação (Retrieval).

        Opções:
        1. 'noop': {self.criteria['noop']}
        2. 'hyde': {self.criteria['hyde']}
        3. 'multi_query': {self.criteria['multi_query']}

        Responda ESTRITAMENTE neste formato JSON:
        {{"strategy": "nome_da_estrategia_escolhida"}}

        Pergunta: "{question}"
        JSON:
        """
