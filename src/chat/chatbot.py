import logging
import numpy as np

# Importações dos componentes
from src.components.llm import LLM
from src.chat.history import ChatHistory
from src.caching.cache_manager import CacheManager
from src.caching.semantic_cache import SemanticCache
from src.pipeline import RAGSystem
from src.preprocessing.query_corrector import QueryCorrector

# Configuração de Logger
logger = logging.getLogger(__name__)


class Chatbot:
    """
    Controlador conversacional que gerencia o fluxo de interação Usuário <-> RAG.

    Responsabilidades:
    1. Gerenciar o histórico da conversa (Memória).
    2. Contextualizar perguntas (Transformar "E ele?" em "Quem é o autor?").
    3. Verificar caches (Exato e Semântico) para respostas rápidas.
    4. Acionar o RAGSystem para obter respostas baseadas em documentos.
    """

    def __init__(self, llm: LLM, rag_system: RAGSystem,
                 cache_manager: CacheManager, semantic_cache: SemanticCache,
                 query_corrector: QueryCorrector):
        """
        Inicializa o Chatbot.

        Args:
            llm (LLM): Usado para a tarefa de contextualização (condensação) da pergunta.
            rag_system (RAGSystem): O motor de busca e resposta.
            cache_manager (CacheManager): Cache L1 (Exato).
            semantic_cache (SemanticCache): Cache L2 (Semântico).
            query_corrector (QueryCorrector): Ferramenta de limpeza e correção ortográfica.
        """
        self.llm = llm
        self.rag_system = rag_system
        self.cache_manager = cache_manager
        self.semantic_cache = semantic_cache
        self.query_corrector = query_corrector

        # Histórico em memória (poderia ser persistido em Redis/DB)
        self.history = ChatHistory()

        # Prompts para Contextualização (Condense Question)
        self.condense_system_prompt = (
            "Você é uma ferramenta de reescrita de consultas. "
            "Sua única tarefa é reescrever a 'Pergunta de Acompanhamento' para que ela "
            "se torne uma pergunta completa e independente, incorporando o contexto necessário "
            "do Histórico. Mantenha o idioma original."
        )
        self.condense_prompt_template = """
        [HISTÓRICO DA CONVERSA]
        {chat_history}

        [PERGUNTA DE ACOMPANHAMENTO]
        {question}

        [PERGUNTA AUTÔNOMA REESCRITA]
        """

    def _condense_question(self, question: str) -> str:
        """
        Transforma uma pergunta dependente de contexto em uma pergunta autônoma (Standalone).

        Exemplo:
        Histórico: "Quem descobriu o Brasil?" -> "Pedro Álvares Cabral".
        Pergunta: "Em que ano?"
        Saída: "Em que ano Pedro Álvares Cabral descobriu o Brasil?"
        """
        # 1. Otimização: Se não há histórico, a pergunta já é autônoma.
        if self.history.is_empty():
            logger.debug("Histórico vazio. Pulando condensação.")
            return question

        chat_history_str = self.history.get_formatted_history()

        # 2. Otimização: Heurística de tamanho (perguntas longas geralmente já têm contexto)
        if len(question.split()) > 10:
            logger.debug("Pergunta longa detectada. Assumindo que é autônoma.")
            return question

        logger.info("Contextualizando pergunta com base no histórico...")
        prompt = self.condense_prompt_template.format(chat_history=chat_history_str, question=question)

        try:
            standalone_question = self.llm.generate_response(
                prompt=prompt,
                system_prompt=self.condense_system_prompt,
                max_new_tokens=150,
                temperature=0.1  # Baixa temperatura para fidelidade
            )

            # Limpeza defensiva de prefixos comuns que LLMs gostam de adicionar
            clean_q = standalone_question.replace("Pergunta Autônoma:", "").replace("Standalone Question:", "").strip()
            logger.info(f"🔄 Pergunta Original: '{question}' | Autônoma: '{clean_q}'")
            return clean_q

        except Exception as e:
            logger.error(f"Falha na condensação: {e}. Usando pergunta original.")
            return question

    def chat(self, user_input: str) -> str:
        """
        Processa a mensagem do usuário ponta a ponta.

        Fluxo: Correção -> Condensação -> Cache -> RAG -> Histórico.
        """
        if not user_input.strip():
            return ""

        logger.info(f"💬 Usuário: {user_input}")

        # --- 1. CORREÇÃO ORTOGRÁFICA ---
        # Usa o método .correct() que definimos no QueryCorrector atualizado
        corrected_input = self.query_corrector.correct(user_input)
        if corrected_input != user_input:
            logger.info(f"✏️ Correção aplicada: '{user_input}' -> '{corrected_input}'")

        # --- 2. CONTEXTUALIZAÇÃO (Condense) ---
        # Transformamos a pergunta em "Standalone" ANTES de checar o cache.
        # Isso garante que "E ele?" busque no cache por "Quem é X?" e não pela string "E ele?".
        standalone_question = self._condense_question(corrected_input)

        # --- 3. VERIFICAÇÃO DE CACHE (L1 e L2) ---
        # Cache Exato (L1)
        cached_response = self.cache_manager.get(standalone_question)
        if cached_response:
            logger.info("🚀 Cache L1 HIT (Exato)")
            self._update_history(user_input, cached_response)
            return cached_response

        # Preparar Embedding para Cache Semântico e RAG (evita gerar 2x)
        # Nota: O RAGSystem gera embeddings internamente, mas para o Cache L2 precisamos aqui.
        # Se performance for crítica, podemos passar esse vetor para o RAGSystem depois.
        try:
            # Pega o primeiro vetor da lista
            query_embedding_list = self.rag_system.embedder.generate_embeddings([standalone_question])
            query_embedding_np = np.array([query_embedding_list[0]], dtype='float32')

            # Cache Semântico (L2)
            cached_response = self.semantic_cache.check(query_embedding_np)
            if cached_response:
                logger.info("🚀 Cache L2 HIT (Semântico)")
                # Atualiza L1 para ficar ainda mais rápido na próxima
                self.cache_manager.set(standalone_question, cached_response)
                self._update_history(user_input, cached_response)
                return cached_response

        except Exception as e:
            logger.warning(f"Erro ao verificar cache semântico (ignorando): {e}")
            query_embedding_np = None

        # --- 4. EXECUÇÃO DO RAG ---
        logger.info("🔍 Cache MISS. Acionando RAG System...")

        # O Chatbot apenas delega a pergunta autônoma.
        # O RAGSystem cuida internamente do Roteamento (Router), HyDE, MultiQuery, etc.
        try:
            rag_result = self.rag_system.ask(standalone_question)
            answer = rag_result['answer']

            # Log da estratégia usada (informativo)
            strategy = rag_result.get('strategy_used', 'Unknown')
            logger.info(f"✅ Resposta gerada via RAG (Estratégia: {strategy})")

        except Exception as e:
            logger.error(f"❌ Erro crítico no RAGSystem: {e}")
            return "Desculpe, encontrei um erro interno ao processar sua solicitação."

        # --- 5. ATUALIZAÇÃO DE CACHE E HISTÓRICO ---
        # Só cacheamos se a resposta for útil (evita cachear "Não sei")
        if answer and "não foi encontrada" not in answer.lower():
            self.cache_manager.set(standalone_question, answer)
            if query_embedding_np is not None:
                self.semantic_cache.add(query_embedding_np, answer)

        self._update_history(user_input, answer)

        return answer

    def _update_history(self, user_msg: str, assistant_msg: str):
        """Helper para salvar no histórico mantendo a interface limpa."""
        self.history.add_message(role="user", content=user_msg)
        self.history.add_message(role="assistant", content=assistant_msg)