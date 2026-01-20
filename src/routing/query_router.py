import logging
import json
from typing import Dict, Optional
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
    1. **Heurística Rápida:** Verifica características simples (tamanho, padrões) para decisão imediata.
    2. **Análise Semântica (LLM):** Envia a pergunta para o LLM com instruções Few-Shot para classificação.

    Attributes:
        llm (LLM): Instância do modelo de linguagem usado para a classificação.
        strategies (Dict[str, QueryTransformer]): Mapeamento de nomes ('noop', 'hyde') para instâncias.
    """

    def __init__(self, llm: LLM, strategies: Dict[str, QueryTransformer]):
        """
        Inicializa o Roteador.

        Args:
            llm (LLM): O cérebro que decidirá a rota em casos complexos.
            strategies (Dict[str, QueryTransformer]): As ferramentas disponíveis.
        """
        self.llm = llm
        self.strategies = strategies

        # Critérios detalhados para guiar o LLM
        self.criteria = {
            "noop": "Use APENAS para perguntas extremamente específicas que contenham identificadores exatos (CNPJ, IDs, Códigos, Logs) e que não precisem de expansão.",
            "hyde": "Use para perguntas complexas, abstratas, pedidos de 'Como funciona', 'Por que', ou definições teóricas que exigem raciocínio.",
            "multi_query": "A MELHOR OPÇÃO para perguntas curtas, factuais ('Qual é...', 'Quem foi...'), geográficas ou vagas. Use sempre que houver sinônimos possíveis."
        }

        logger.info(f"📍 QueryRouter inicializado com estratégias: {list(self.strategies.keys())}")

    def route(self, question: str) -> QueryTransformer:
        """
        Seleciona a estratégia de transformação mais adequada para a pergunta.

        Atua como um orquestrador (facade), delegando a decisão para heurísticas
        ou para o LLM, e retornando o objeto transformador correspondente.

        Args:
            question (str): A pergunta original do usuário.

        Returns:
            QueryTransformer: A instância da estratégia escolhida. Retorna 'noop' em caso de falha.
        """
        # 1. Tenta decisão rápida (Heurística)
        heuristic_choice = self._check_heuristics(question)
        if heuristic_choice:
            logger.info(f"⚡ Roteador (Heurística): Decisão rápida tomada -> '{heuristic_choice.upper()}'")
            return self.strategies.get(heuristic_choice, self.strategies.get("noop"))

        # 2. Tenta decisão profunda (LLM)
        llm_choice = self._decide_via_llm(question)

        logger.info(f"✅ Roteador (LLM): Decisão final -> '{llm_choice.upper()}' para '{question[:30]}...'")
        return self.strategies.get(llm_choice, self.strategies.get("noop"))

    def _check_heuristics(self, question: str) -> Optional[str]:
        """
        Aplica regras determinísticas baseadas na estrutura da string.

        Args:
            question (str): A pergunta do usuário.

        Returns:
            Optional[str]: O nome da estratégia ('multi_query', etc) ou None se nenhuma regra se aplicar.
        """
        # Regra 1: Perguntas muito curtas (< 4 palavras) geralmente precisam de expansão de contexto.
        # Ex: "Capital do Brasil" -> Precisa virar "Qual a capital..." / "Cidade capital..."
        if len(question.split()) < 4:
            return "multi_query"

        return None

    def _decide_via_llm(self, question: str) -> str:
        """
        Consulta o LLM para classificar a intenção da pergunta.

        Args:
            question (str): A pergunta do usuário.

        Returns:
            str: A chave da estratégia escolhida (ex: 'hyde'). Retorna 'noop' em caso de erro.
        """
        logger.debug("🤔 Roteador (LLM): Analisando intenção da pergunta...")
        prompt = self._build_prompt(question)

        try:
            # Solicita uma resposta curta e determinística
            response = self.llm.generate_response(
                prompt=prompt,
                system_prompt="Você é um classificador de intenção de busca (RAG Router).",
                max_new_tokens=10,
                temperature=0.0
            )

            # Delega a limpeza e validação da string
            return self._parse_strategy_key(response)

        except Exception as e:
            logger.error(f"❌ Erro genérico no Roteador (LLM): {e}. Usando fallback 'noop'.")
            return "noop"

    def _parse_strategy_key(self, raw_response: str) -> str:
        """
        Limpa a resposta bruta do LLM e mapeia para uma chave válida.

        Args:
            raw_response (str): O texto retornado pelo LLM (ex: " 'multi_query' ", "hyde.", etc).

        Returns:
            str: Uma chave válida presente em `self.strategies` ou 'noop' (fallback).
        """
        # Normalização básica
        cleaned = raw_response.strip().lower()
        cleaned = cleaned.replace("'", "").replace('"', "").replace(".", "")

        # 1. Match Exato
        if cleaned in self.strategies:
            return cleaned

        # 2. Match Parcial (Fuzzy) - Caso o LLM seja verboso
        if "multi" in cleaned or "query" in cleaned:
            return "multi_query"
        elif "hyde" in cleaned:
            return "hyde"
        elif "noop" in cleaned or "exata" in cleaned:
            return "noop"

        logger.warning(f"⚠️ Roteador retornou chave desconhecida: '{raw_response}'. Usando fallback 'noop'.")
        return "noop"

    def _build_prompt(self, question: str) -> str:
        """
        Constrói o prompt de classificação usando Few-Shot Learning (Exemplos).

        Args:
            question (str): A pergunta a ser analisada.

        Returns:
            str: O prompt formatado.
        """
        return f"""
        Analise a PERGUNTA DO USUÁRIO e classifique-a em uma das seguintes estratégias de busca:

        1. 'noop': {self.criteria['noop']}
        2. 'hyde': {self.criteria['hyde']}
        3. 'multi_query': {self.criteria['multi_query']}

        EXEMPLOS PARA GUIAR SUA DECISÃO:
        - "Erro 500 no endpoint /login" -> noop
        - "Qual o CPF do cliente 9988?" -> noop
        - "Explique o impacto da inflação nos juros" -> hyde
        - "Como funciona a fotossíntese?" -> hyde
        - "Capital do Brasil" -> multi_query
        - "Qual o maior pico brasileiro?" -> multi_query (Fato geográfico/Sinônimos)
        - "Melhores praias do nordeste" -> multi_query

        PERGUNTA DO USUÁRIO: "{question}"

        Retorne APENAS o nome da estratégia (noop, hyde ou multi_query). Nada mais.
        Resposta:
        """