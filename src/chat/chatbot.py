from typing import Dict

from src.components.llm import LLM
from src.chat.history import ChatHistory
from src.caching.cache_manager import CacheManager
from src.caching.semantic_cache import SemanticCache
from ..routing.query_router import QueryRouter
from ..query_transformers import QueryTransformer
from src.pipeline import RAGSystem
import numpy as np

class Chatbot:
    """
    Orquestra a lógica de um chatbot conversacional, utilizando um pipeline RAG como
    sua base de conhecimento.
    """

    def __init__(self, llm: LLM, rag_system: RAGSystem, cache_manager: CacheManager, semantic_cache: SemanticCache, transformers: Dict[str, QueryTransformer]):
        self.llm = llm
        self.rag_system = rag_system
        self.history = ChatHistory()
        self.cache_manager = cache_manager       # <-- Cache Camada 1
        self.semantic_cache = semantic_cache   # <-- Cache Camada 2
        # O chatbot usa um roteador e um dicionário de transformadores
        self.router = QueryRouter(llm)
        self.transformers = transformers

        self.condense_system_prompt = """Dada uma conversa e uma pergunta de acompanhamento, reescreva a pergunta de acompanhamento para ser uma pergunta autônoma, em sua língua original. Se a pergunta já for autônoma, apenas a retorne."""
        self.condense_prompt_template = """
        Histórico da Conversa:
        {chat_history}

        Pergunta de Acompanhamento: {question}

        Pergunta Autônoma:
        """

    def _condense_question(self, question: str) -> str:
        """
        Usa um LLM para condensar o histórico e a nova pergunta em uma pergunta autônoma.
        """
        chat_history_str = self.history.get_formatted_history()

        # Se a pergunta parecer já ser completa, podemos pular a condensação
        # (Esta é uma otimização simples, pode ser mais elaborada)
        if len(question.split()) > 10:
            print("INFO: Pergunta parece ser autônoma, pulando condensação.")
            return question

        prompt = self.condense_prompt_template.format(chat_history=chat_history_str, question=question)

        print("INFO: Condensando pergunta com o histórico...")
        standalone_question = self.llm.generate_response(
            prompt=prompt,
            system_prompt=self.condense_system_prompt,
            max_new_tokens=100,
            temperature=0.1
        )

        # LIMPAR O PREFIXO
        prefix_to_remove = "Pergunta Autônoma:"
        if standalone_question.startswith(prefix_to_remove):
            standalone_question = standalone_question[len(prefix_to_remove):].strip()

        print(f"INFO: Pergunta Autônoma Gerada: '{standalone_question}'")

        return standalone_question

    def chat(self, user_input: str) -> str:
        """
        Processa uma interação, agora com o funil de cache de duas camadas.
        """
        # --- CAMADA 1: VERIFICAÇÃO DO CACHE EXATO ---
        cached_response = self.cache_manager.get(user_input)
        if cached_response:
            print("INFO: Cache HIT! (Camada 1 - Exato)")
            self.history.add_message(role="user", content=user_input)
            self.history.add_message(role="assistant", content=cached_response)
            return cached_response

        # --- CAMADA 2: VERIFICAÇÃO DO CACHE SEMÂNTICO ---
        query_embedding = self.rag_system.embedder.generate_embeddings([user_input])[0]
        query_embedding_np = np.array([query_embedding], dtype='float32')

        cached_response = self.semantic_cache.check(query_embedding_np)
        if cached_response:
            print("INFO: Cache HIT! (Camada 2 - Semântico)")
            # Adiciona ao histórico
            self.history.add_message(role="user", content=user_input)
            self.history.add_message(role="assistant", content=cached_response)
            # **Importante**: Popula o cache de Camada 1 para futuras buscas mais rápidas
            self.cache_manager.set(user_input, cached_response)
            return cached_response

        # --- CACHE MISS (AMBAS AS CAMADAS) ---
        print("INFO: Cache MISS. Executando o pipeline RAG completo.")

        self.history.add_message(role="user", content=user_input)

        standalone_question = self._condense_question(user_input)
        rag_response = self.rag_system.ask(standalone_question)
        answer = rag_response['answer']

        # --- ATUALIZAÇÃO DOS CACHES ---
        if answer and "não foi encontrada" not in answer:
            print("INFO: Adicionando nova resposta aos caches (Camada 1 e 2).")
            # Adiciona a nova resposta a ambos os caches
            self.cache_manager.set(user_input, answer)
            self.semantic_cache.add(query_embedding_np, answer)

        self.history.add_message(role="assistant", content=answer)

        return answer