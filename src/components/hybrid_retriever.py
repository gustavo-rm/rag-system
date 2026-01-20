import logging
import string
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.stores.base import VectorStore

# Configuração de Logger
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Orquestrador de Busca Híbrida (Dense + Sparse).

    Combina a busca semântica (VectorStore) com a busca por palavras-chave (BM25).
    Esta abordagem mitiga as fraquezas de cada método isolado:
    - Vector Search: Bom para conceitos, ruim para termos exatos/raros.
    - BM25: Bom para termos exatos (IDs, nomes próprios), ruim para sinônimos.

    Attributes:
        vector_store (VectorStore): Instância do banco vetorial persistente.
        bm25 (Optional[BM25Okapi]): Índice em memória para busca de palavras-chave.
        documents_cache (List[Dict]): Cópia local dos metadados necessária para o BM25.
    """

    def __init__(self, vector_store: VectorStore):
        """
        Inicializa o HybridRetriever e tenta sincronizar com dados existentes.

        Args:
            vector_store (VectorStore): Banco de dados vetorial já instanciado.
        """
        self.vector_store = vector_store
        self.bm25: Optional[BM25Okapi] = None
        self.documents_cache: List[Dict[str, Any]] = []

        # Tenta carregar dados pré-existentes para não começar com o BM25 vazio
        self._try_sync_initial_data()

    def _tokenize(self, text: str) -> List[str]:
        """
        Realiza a tokenização do texto para o algoritmo BM25.

        Processo:
        1. Converte para minúsculas.
        2. Substitui pontuações por espaços (evita aglutinação).
        3. Divide por espaços em branco.

        Args:
            text (str): O texto cru a ser processado.

        Returns:
            List[str]: Lista de tokens limpos.

        Example:
            >>> _tokenize("Olá, Brasil!")
            ['olá', 'brasil']
        """
        if not text:
            return []

        # Converte para minúsculas
        text = text.lower()

        # Cria tabela de tradução: Pontuação -> Espaço
        # Isso garante que "fim.inicio" vire "fim inicio" e não "fiminicio"
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        cleaned_text = text.translate(translator)

        return cleaned_text.split()

    def _try_sync_initial_data(self) -> None:
        """
        Sincroniza o índice BM25 (RAM) com os documentos persistidos no VectorStore (Disco).

        Este método é crítico para garantir que, ao reiniciar a aplicação,
        a busca por palavras-chave funcione nos documentos já indexados.

        Raises:
            Exception: Captura e loga falhas de conexão com o banco, sem travar a inicialização.
        """
        # Verifica duck-typing ou atributo específico do ChromaDB
        if hasattr(self.vector_store, 'collection'):
            logger.info("🔄 Sincronizando índice BM25 com dados do VectorStore...")
            try:
                # Otimização: Requisita apenas IDs, Documentos e Metadatas (ignora embeddings pesados)
                all_data = self.vector_store.collection.get(include=['documents', 'metadatas'])

                if all_data and all_data['ids']:
                    ids = all_data['ids']
                    texts = all_data['documents']
                    metadatas = all_data['metadatas']

                    self.documents_cache = []

                    # Reconstrói a estrutura de cache local
                    for doc_id, text, meta in zip(ids, texts, metadatas):
                        if meta is None:
                            meta = {}
                        # Garante que o texto esteja no metadata para acesso rápido
                        meta['text'] = text

                        self.documents_cache.append({
                            'id': doc_id,
                            'metadata': meta
                        })

                    self._rebuild_bm25()
                    logger.info(f"✅ BM25 reconstruído com sucesso ({len(self.documents_cache)} documentos).")
                else:
                    logger.info("ℹ️ VectorStore vazio. BM25 iniciará vazio.")

            except Exception as e:
                logger.warning(f"⚠️ Falha não-crítica ao sincronizar BM25: {e}")

    def _rebuild_bm25(self) -> None:
        """
        Recalcula o índice BM25 completo com base no `documents_cache` atual.
        Deve ser chamado sempre que novos documentos são adicionados.
        """
        if not self.documents_cache:
            return

        # Aplica a tokenização corrigida em todo o corpus
        tokenized_corpus = [
            self._tokenize(doc['metadata'].get('text', ''))
            for doc in self.documents_cache
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def add_documents(self, chunks: List[str], embeddings: List[List[float]],
                      metadatas: Optional[List[Dict]] = None) -> None:
        """
        Adiciona novos documentos ao sistema (Atualiza VectorStore e BM25).

        Args:
            chunks (List[str]): Lista de textos dos documentos.
            embeddings (List[List[float]]): Vetores gerados pelo Embedder.
            metadatas (Optional[List[Dict]]): Metadados opcionais (fonte, página, etc).
        """
        import time

        # Gera IDs únicos baseados em timestamp para evitar colisões
        start_ts = int(time.time() * 1000)
        ids = [str(start_ts + i) for i in range(len(chunks))]

        # 1. Persistência no Vector DB
        self.vector_store.store_embeddings(chunks, embeddings, ids=ids, metadatas=metadatas)

        # 2. Atualização do BM25 (Memória)
        for i, chunk in enumerate(chunks):
            meta = metadatas[i] if metadatas else {}
            meta['text'] = chunk

            self.documents_cache.append({
                'id': ids[i],
                'metadata': meta
            })

        self._rebuild_bm25()
        logger.info(f"➕ HybridRetriever: {len(chunks)} novos documentos indexados.")

    def search(self, query_text: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executa a busca híbrida (Fusão de Resultados).

        Estratégia:
        1. Busca top_k via Vetores (Similaridade Semântica).
        2. Busca top_k via BM25 (Similaridade Lexical/Palavra-chave).
        3. Realiza a fusão dos resultados usando o ID do documento como chave de deduplicação.

        Args:
            query_text (str): A pergunta em linguagem natural (para o BM25).
            query_embedding (List[float]): O vetor da pergunta (para o VectorStore).
            top_k (int): Quantidade de documentos a recuperar de CADA fonte.

        Returns:
            List[Dict[str, Any]]: Lista unificada e desduplicada de documentos encontrados.
        """
        # A. Busca Vetorial (Dense)
        vector_results = self.vector_store.search(query_embedding, top_k=top_k)

        # Normalização dos resultados vetoriais
        for doc in vector_results:
            doc['source'] = 'vector'
            if 'score' not in doc:
                doc['score'] = 0.0

        # B. Busca BM25 (Sparse)
        bm25_results = []
        if self.bm25:
            tokenized_query = self._tokenize(query_text)

            # get_top_n retorna os itens crus do cache (dicts com id e metadata)
            top_docs_bm25 = self.bm25.get_top_n(tokenized_query, self.documents_cache, n=top_k)

            for doc in top_docs_bm25:
                bm25_results.append({
                    'id': doc['id'],
                    'metadata': doc['metadata'],
                    'page_content': doc['metadata'].get('text', ''),
                    'score': 0.0,  # BM25Okapi (rank_bm25) não expõe score facilmente aqui, placeholder para ReRanker
                    'source': 'bm25'
                })

        # C. Fusão e Deduplicação (Chave: ID)
        combined_docs: Dict[str, Dict[str, Any]] = {}

        # 1. Prioridade para Vetorial
        for doc in vector_results:
            doc_id = doc.get('id')
            if doc_id:
                combined_docs[doc_id] = doc

        # 2. Complemento com BM25
        for doc in bm25_results:
            doc_id = doc.get('id')
            if doc_id:
                if doc_id not in combined_docs:
                    combined_docs[doc_id] = doc
                else:
                    # Se já existe, marca como híbrido (encontrado pelos dois métodos)
                    combined_docs[doc_id]['source'] = 'hybrid'

        final_results = list(combined_docs.values())

        logger.info(
            f"🔎 Busca Híbrida: {len(vector_results)} (Vector) + {len(bm25_results)} (BM25) -> {len(final_results)} Únicos")

        return final_results