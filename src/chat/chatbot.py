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

        # Histórico em memória
        self.history = ChatHistory(max_history_len=6)

        # --- CONFIGURAÇÃO DO PROMPT DE CONTEXTUALIZAÇÃO (BLINDADO) ---
        self.context_system_prompt = "Você é um assistente especialista em reescrever perguntas para sistemas de busca."

        # Este prompt contém as "Guardrails" para evitar alucinação de contexto
        self.context_prompt_template = """
        Histórico da Conversa:
        {chat_history}

        Última Pergunta do Usuário: "{question}"

        Você é um módulo de REESCRITA DE PERGUNTAS.
        Você NÃO responde perguntas.
        Você APENAS reescreve a 'Última Pergunta', seguindo rigorosamente as regras abaixo.
        
        ==============================
        ORDEM DE PRIORIDADE (SIGA ESTRITAMENTE)
        ==============================
        1. NUNCA responda à pergunta.
        2. Se a pergunta for clara e independente do histórico, REPITA-A EXATAMENTE como está.
        3. Use o histórico SOMENTE para substituir pronomes ambíguos.
        4. Se houver mudança de tópico, IGNORE completamente o histórico.
        
        ==============================
        DEFINIÇÃO DE NOVO TÓPICO
        ==============================
        Considere que há NOVO TÓPICO quando o substantivo principal da pergunta muda
        em relação ao histórico anterior.
        
        Exemplo:
        Histórico: "Qual é a capital do Brasil?"
        Última Pergunta: "E o relevo?"
        → Novo tópico → ignore o histórico.
        
        ==============================
        REGRAS OBRIGATÓRIAS
        ==============================
        - NÃO adicione informações.
        - NÃO reformule estilo, tom ou vocabulário.
        - NÃO torne a pergunta mais específica.
        - NÃO infira intenções ocultas.
        - NÃO conecte ideias que não estejam explicitamente na pergunta.
        
        ==============================
        REGRAS DE REESCRITA
        ==============================
        - Se a pergunta usar pronomes ("ele", "ela", "isso", "aquilo") referindo-se ao histórico,
        substitua APENAS pelo termo correto já presente no histórico.
        - Se NÃO houver pronomes ambíguos, REPITA a pergunta exatamente,
        mantendo todas as palavras, pontuação e ordem.
        
        ==============================
        EXEMPLOS
        ==============================
        
        Exemplo 1 — Pergunta clara (sem depender do histórico)
        Última Pergunta:
        "Qual o maior pico do Brasil?"
        
        Saída:
        "Qual o maior pico do Brasil?"
        
        ------------------------------
        
        Exemplo 2 — Uso de pronome dependente do histórico
        Histórico:
        "Qual é a capital do Brasil?"
        
        Última Pergunta:
        "E qual é a população dela?"
        
        Saída:
        "Qual é a população da capital do Brasil?"
        
        ------------------------------
        
        Exemplo 3 — Novo tópico (histórico ignorado)
        Histórico:
        "Qual é a capital do Brasil?"
        
        Última Pergunta:
        "Como é o relevo?"
        
        Saída:
        "Como é o relevo?"
        
        ==============================
        FORMATO DA SAÍDA
        ==============================
        - Retorne APENAS a pergunta final.
        - NÃO inclua explicações, comentários ou qualquer outro texto.
        
        Pergunta Reformulada (apenas o texto):
        """

    def _contextualize_question(self, question: str) -> str:
        """
        Transforma uma pergunta dependente de contexto em uma pergunta autônoma (Standalone),
        com proteção contra mudança de tópico.

        Exemplo 1 (Continuação):
        Histórico: "Quem descobriu o Brasil?" -> "Cabral".
        Pergunta: "Em que ano?"
        Saída: "Em que ano Cabral descobriu o Brasil?"

        Exemplo 2 (Mudança de Tópico):
        Histórico: "Quem descobriu o Brasil?" -> "Cabral".
        Pergunta: "Qual a capital da França?"
        Saída: "Qual a capital da França?" (Histórico ignorado)
        """
        # 1. Otimização: Se não há histórico, a pergunta já é autônoma.
        if self.history.is_empty():
            return question

        # 2. Otimização: Perguntas longas (>10 palavras) geralmente já contêm o contexto necessário.
        if len(question.split()) > 10:
            logger.debug("Pergunta longa detectada. Assumindo que é autônoma.")
            return question

        logger.info("Contextualizando pergunta com base no histórico...")

        chat_history_str = self.history.get_formatted_history()
        prompt = self.context_prompt_template.format(chat_history=chat_history_str, question=question)

        try:
            standalone_question = self.llm.generate_response(
                prompt=prompt,
                system_prompt=self.context_system_prompt,
                max_new_tokens=100,  # Curto para evitar divagação
                temperature=0.2  # Baixa temperatura para seguir as regras estritamente
            )

            # Limpeza defensiva de prefixos comuns
            clean_q = standalone_question.strip().replace('Pergunta Reformulada:', '').replace('"', '')

            logger.info(f"🔄 Pergunta Original: '{question}' | Autônoma: '{clean_q}'")
            return clean_q

        except Exception as e:
            logger.error(f"Falha na contextualização: {e}. Usando pergunta original.")
            return question

    def chat(self, user_input: str) -> str:
        """
        Processa a mensagem do usuário ponta a ponta.
        Fluxo: Correção -> Contextualização -> Cache -> RAG -> Histórico.
        """
        if not user_input.strip():
            return ""

        logger.info(f"💬 Usuário: {user_input}")

        # --- 1. CORREÇÃO ORTOGRÁFICA ---
        corrected_input = self.query_corrector.correct_query(user_input)
        if corrected_input != user_input:
            logger.info(f"✏️ Correção aplicada: '{user_input}' -> '{corrected_input}'")

        # --- 2. CONTEXTUALIZAÇÃO ---
        standalone_question = self._contextualize_question(corrected_input)

        # --- 3. VERIFICAÇÃO DE CACHE (L1 e L2) ---
        cached_response = self.cache_manager.get(standalone_question)
        if cached_response:
            logger.info("🚀 Cache L1 HIT (Exato)")
            self._update_history(user_input, cached_response)
            return cached_response

        # Preparar Embedding para Cache Semântico
        try:
            query_embedding_list = self.rag_system.embedder.generate_embeddings([standalone_question])
            query_embedding_np = np.array([query_embedding_list[0]], dtype='float32')

            cached_response = self.semantic_cache.check(query_embedding_np)
            if cached_response:
                logger.info("🚀 Cache L2 HIT (Semântico)")
                self.cache_manager.set(standalone_question, cached_response)
                self._update_history(user_input, cached_response)
                return cached_response

        except Exception as e:
            logger.warning(f"Erro ao verificar cache semântico (ignorando): {e}")
            query_embedding_np = None

        # --- 4. EXECUÇÃO DO RAG ---
        logger.info("🔍 Cache MISS. Acionando RAG System...")

        try:
            rag_result = self.rag_system.ask(standalone_question, rerank_top_n=5)
            answer = rag_result['answer']
            strategy = rag_result.get('strategy_used', 'Unknown')
            logger.info(f"✅ Resposta gerada via RAG (Estratégia: {strategy})")

        except Exception as e:
            logger.error(f"❌ Erro crítico no RAGSystem: {e}")
            return "Desculpe, encontrei um erro interno ao processar sua solicitação."

        # --- 5. ATUALIZAÇÃO DE CACHE E HISTÓRICO ---
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