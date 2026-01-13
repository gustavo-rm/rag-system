import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from src.stores.base import VectorStore


class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        """
        Gerencia a busca híbrida (BM25 + Vetorial).
        Recebe uma instância já configurada de VectorStore (Chroma ou Pinecone).
        """
        self.vector_store = vector_store
        self.bm25 = None
        self.documents_cache = []  # Cache local de textos para o BM25

        # Tenta sincronizar dados existentes se for Chroma (local)
        # Para Pinecone, isso pode ser lento dependendo do tamanho, então deixamos opcional
        self._try_sync_initial_data()

    def _try_sync_initial_data(self):
        """
        Tenta puxar dados do banco para reconstruir o índice BM25 na inicialização.
        Isso é crucial para não perder a capacidade de busca por palavras-chave ao reiniciar o app.
        """
        # Verifica se é ChromaStore olhando o nome da classe ou atributo
        if "ChromaStore" in str(type(self.vector_store)):
            print("🔄 Sincronizando BM25 com dados do ChromaDB...")
            try:
                # O método get() do Chroma retorna tudo se não passarmos filtros
                all_data = self.vector_store.collection.get()
                if all_data and all_data['documents']:
                    texts = all_data['documents']
                    ids = all_data['ids']
                    # Reconstrói a estrutura interna
                    self.documents_cache = [{'id': i, 'metadata': {'text': t}} for i, t in zip(ids, texts)]
                    self._rebuild_bm25()
                    print(f"✅ BM25 reconstruído com {len(texts)} documentos existentes.")
            except Exception as e:
                print(f"⚠️ Erro ao sincronizar ChromaDB: {e}")

    def _rebuild_bm25(self):
        """Recria o índice BM25 com os documentos atuais do cache."""
        if not self.documents_cache:
            return

        tokenized_corpus = [doc['metadata']['text'].lower().split(" ") for doc in self.documents_cache]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def add_documents(self, chunks: List[str], embeddings: List[List[float]]):
        """
        Adiciona documentos ao VectorStore E atualiza o índice BM25.
        Substitui a chamada direta de 'store_embeddings'.
        """
        # 1. Salva no Banco Vetorial (Persistência)
        # Gera IDs simples baseados no timestamp ou sequencial se não existirem
        import time
        start_id = int(time.time() * 1000)
        ids = [str(start_id + i) for i in range(len(chunks))]

        self.vector_store.store_embeddings(chunks, embeddings, ids=ids)

        # 2. Atualiza índice BM25 (Memória)
        new_docs = [{'id': i, 'metadata': {'text': c}} for i, c in zip(ids, chunks)]
        self.documents_cache.extend(new_docs)
        self._rebuild_bm25()
        print(f"➕ HybridRetriever: {len(chunks)} documentos indexados (BM25 + Vetorial).")

    def search(self, query_text: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executa a busca híbrida e combina os resultados.
        """
        # --- A. Busca Vetorial (Via VectorStore existente) ---
        vector_results = self.vector_store.search(query_embedding, top_k=top_k)

        # --- B. Busca BM25 (Local) ---
        bm25_results = []
        if self.bm25:
            tokenized_query = query_text.lower().split(" ")
            # O BM25Okapi retorna os documentos, mas precisamos formatar igual ao vector store
            # Pegamos top_k * 2 para garantir variedade antes da fusão
            top_docs = self.bm25.get_top_n(tokenized_query, self.documents_cache, n=top_k)

            for doc in top_docs:
                # BM25 não dá score normalizado fácil, então marcamos como origem 'bm25'
                # para prioridade ou usamos um score fictício alto se for match exato
                bm25_results.append({
                    'id': doc['id'],
                    'score': 0.0,  # Score placeholder, o ReRanker vai corrigir isso depois!
                    'metadata': doc['metadata'],
                    'source': 'bm25'
                })

        # --- C. Fusão e Deduplicação ---
        # Usamos um dicionário para remover duplicatas por ID ou Conteúdo
        combined_docs = {}

        # Adiciona vetoriais
        for doc in vector_results:
            doc['source'] = 'vector'
            combined_docs[doc['metadata']['text']] = doc

        # Adiciona BM25 (sobrescreve se já existir? Não, mantemos o vetor que tem score real)
        for doc in bm25_results:
            if doc['metadata']['text'] not in combined_docs:
                combined_docs[doc['metadata']['text']] = doc

        return list(combined_docs.values())
