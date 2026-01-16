from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from src.stores.base import VectorStore
import logging
# Configuração de Logger
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Orquestrador de Busca Híbrida.

    Combina a precisão de palavras-chave do algoritmo BM25 (Sparse) com a
    compreensão semântica da Busca Vetorial (Dense). Mantém o índice BM25 em memória
    e sincroniza com o VectorStore persistente.
    """

    def __init__(self, vector_store: VectorStore):
        """
        Inicializa o Retriever.

        Args:
            vector_store (VectorStore): Instância configurada do banco vetorial (Chroma, Pinecone, etc).
        """
        self.vector_store = vector_store
        self.bm25 = None
        self.documents_cache = []  # Cache local necessário pois BM25 roda em RAM

        # Tenta carregar dados existentes do banco para não perder o índice BM25 ao reiniciar
        self._try_sync_initial_data()

    def _try_sync_initial_data(self):
        """Tenta recuperar documentos do VectorStore para reconstruir o índice BM25 na inicialização."""
        if "ChromaStore" in str(type(self.vector_store)):
            logger.info("🔄 Sincronizando índice BM25 com dados persistidos no ChromaDB...")
            try:
                all_data = self.vector_store.collection.get()
                if all_data and all_data['documents']:
                    texts = all_data['documents']
                    ids = all_data['ids']
                    self.documents_cache = [{'id': i, 'metadata': {'text': t}} for i, t in zip(ids, texts)]
                    self._rebuild_bm25()
                    logger.info(f"✅ BM25 reconstruído com sucesso ({len(texts)} docs).")
            except Exception as e:
                logger.error(f"⚠️ Aviso: Falha ao sincronizar ChromaDB: {e}")

    def _rebuild_bm25(self):
        """(Re)cria o índice BM25 usando os documentos atuais do cache."""
        if not self.documents_cache:
            return
        # Tokenização simples por espaço para o BM25
        tokenized_corpus = [doc['metadata']['text'].lower().split(" ") for doc in self.documents_cache]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def add_documents(self, chunks: List[str], embeddings: List[List[float]]):
        """
        Adiciona novos documentos ao sistema completo.

        1. Persiste os embeddings no VectorStore.
        2. Atualiza o cache local e reconstrói o índice BM25.

        Args:
            chunks (List[str]): Lista de textos.
            embeddings (List[List[float]]): Lista de vetores correspondentes.
        """
        import time
        start_id = int(time.time() * 1000)
        ids = [str(start_id + i) for i in range(len(chunks))]

        # 1. Armazenamento Vetorial
        self.vector_store.store_embeddings(chunks, embeddings, ids=ids)

        # 2. Indexação BM25 (Memória)
        new_docs = [{'id': i, 'metadata': {'text': c}} for i, c in zip(ids, chunks)]
        self.documents_cache.extend(new_docs)
        self._rebuild_bm25()
        logger.info(f"➕ HybridRetriever: {len(chunks)} novos documentos indexados.")

    def search(self, query_text: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executa a busca em ambos os índices e funde os resultados.

        Args:
            query_text (str): A pergunta em texto (para o BM25).
            query_embedding (List[float]): O vetor da pergunta (para o VectorStore).
            top_k (int): Número de documentos a recuperar de CADA fonte.

        Returns:
            List[Dict[str, Any]]: Lista desduplicada de documentos candidatos.
        """
        # A. Busca Vetorial
        vector_results = self.vector_store.search(query_embedding, top_k=top_k)

        # B. Busca BM25
        bm25_results = []
        if self.bm25:
            tokenized_query = query_text.lower().split(" ")
            top_docs = self.bm25.get_top_n(tokenized_query, self.documents_cache, n=top_k)
            for doc in top_docs:
                bm25_results.append({
                    'id': doc['id'],
                    'score': 0.0,  # Placeholder, será corrigido pelo ReRanker
                    'metadata': doc['metadata'],
                    'source': 'bm25'
                })

        # C. Fusão e Deduplicação (Prioriza Vector se houver conflito, ou mantém ambos)
        combined_docs = {}
        for doc in vector_results:
            doc['source'] = 'vector'
            combined_docs[doc['metadata']['text']] = doc  # Usa o texto como chave de unicidade

        for doc in bm25_results:
            if doc['metadata']['text'] not in combined_docs:
                combined_docs[doc['metadata']['text']] = doc

        return list(combined_docs.values())