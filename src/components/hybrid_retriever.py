import logging
import string
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.stores.base import VectorStore

# Logger Configuration
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid Search Orchestrator (Dense + Sparse).

    Combines semantic search (VectorStore) with keyword search (BM25).
    This approach mitigates the weaknesses of each isolated method:
    - Vector Search: Good for concepts, bad for exact/rare terms.
    - BM25: Good for exact terms (IDs, proper names), bad for synonyms.

    Attributes:
        vector_store (VectorStore): Instance of the persistent vector database.
        bm25 (Optional[BM25Okapi]): In-memory index for keyword search.
        documents_cache (List[Dict]): Local copy of metadata needed for BM25.
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initializes the HybridRetriever and attempts to synchronize with existing data.

        Args:
            vector_store (VectorStore): Already instantiated vector database.
        """
        self.vector_store = vector_store
        self.bm25: Optional[BM25Okapi] = None
        self.documents_cache: List[Dict[str, Any]] = []

        # Tries to load pre-existing data so BM25 doesn't start empty
        self._try_sync_initial_data()

    def _tokenize(self, text: str) -> List[str]:
        """
        Performs tokenization of the text for the BM25 algorithm.

        Process:
        1. Converts to lowercase.
        2. Replaces punctuation with spaces (avoids agglutination).
        3. Splits by whitespace.

        Args:
            text (str): The raw text to be processed.

        Returns:
            List[str]: List of clean tokens.

        Example:
            >>> _tokenize("Hello, World!")
            ['hello', 'world']
        """
        if not text:
            return []

        # Converts to lowercase
        text = text.lower()

        # Creates translation table: Punctuation -> Space
        # This ensures "end.start" becomes "end start" and not "endstart"
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        cleaned_text = text.translate(translator)

        return cleaned_text.split()

    def _try_sync_initial_data(self) -> None:
        """
        Synchronizes the BM25 index (RAM) with documents persisted in the VectorStore (Disk).

        This method is critical to ensure that, upon restarting the application,
        keyword search works on already indexed documents.

        Raises:
            Exception: Captures and logs database connection failures without crashing initialization.
        """
        # Checks for duck-typing or specific ChromaDB attribute
        if hasattr(self.vector_store, 'collection'):
            logger.info("🔄 Synchronizing BM25 index with VectorStore data...")
            try:
                # Optimization: Requests only IDs, Documents, and Metadatas (ignores heavy embeddings)
                all_data = self.vector_store.collection.get(include=['documents', 'metadatas'])

                if all_data and all_data['ids']:
                    ids = all_data['ids']
                    texts = all_data['documents']
                    metadatas = all_data['metadatas']

                    self.documents_cache = []

                    # Reconstructs local cache structure
                    for doc_id, text, meta in zip(ids, texts, metadatas):
                        if meta is None:
                            meta = {}
                        # Ensures text is in metadata for quick access
                        meta['text'] = text

                        self.documents_cache.append({
                            'id': doc_id,
                            'metadata': meta
                        })

                    self._rebuild_bm25()
                    logger.info(f"✅ BM25 successfully rebuilt ({len(self.documents_cache)} documents).")
                else:
                    logger.info("ℹ️ VectorStore empty. BM25 will start empty.")

            except Exception as e:
                logger.warning(f"⚠️ Non-critical failure synchronizing BM25: {e}")

    def _rebuild_bm25(self) -> None:
        """
        Recalculates the complete BM25 index based on the current `documents_cache`.
        Must be called whenever new documents are added.
        """
        if not self.documents_cache:
            return

        # Applies corrected tokenization across the entire corpus
        tokenized_corpus = [
            self._tokenize(doc['metadata'].get('text', ''))
            for doc in self.documents_cache
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def add_documents(self, chunks: List[str], embeddings: List[List[float]],
                      metadatas: Optional[List[Dict]] = None) -> None:
        """
        Adds new documents to the system (Updates VectorStore and BM25).

        Args:
            chunks (List[str]): List of document texts.
            embeddings (List[List[float]]): Vectors generated by the Embedder.
            metadatas (Optional[List[Dict]]): Optional metadata (source, page, etc.).
        """
        import time

        # Generates unique IDs based on timestamp to avoid collisions
        start_ts = int(time.time() * 1000)
        ids = [str(start_ts + i) for i in range(len(chunks))]

        # 1. Persistence in Vector DB
        self.vector_store.store_embeddings(chunks, embeddings, ids=ids, metadatas=metadatas)

        # 2. Update BM25 (Memory)
        for i, chunk in enumerate(chunks):
            meta = metadatas[i] if metadatas else {}
            meta['text'] = chunk

            self.documents_cache.append({
                'id': ids[i],
                'metadata': meta
            })

        self._rebuild_bm25()
        logger.info(f"➕ HybridRetriever: {len(chunks)} new documents indexed.")

    def search(self, query_text: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes hybrid search (Result Fusion).

        Strategy:
        1. Top_k via Vectors (Semantic Similarity).
        2. Top_k via BM25 (Lexical/Keyword Similarity).
        3. Fuses results using document ID as deduplication key.

        Args:
            query_text (str): The natural language question (for BM25).
            query_embedding (List[float]): The question vector (for VectorStore).
            top_k (int): Number of documents to retrieve from EACH source.

        Returns:
            List[Dict[str, Any]]: Unified and deduplicated list of found documents.
        """
        # A. Vector Search (Dense)
        vector_results = self.vector_store.search(query_embedding, top_k=top_k)

        # Normalization of vector results
        for doc in vector_results:
            doc['source'] = 'vector'
            if 'score' not in doc:
                doc['score'] = 0.0

        # B. BM25 Search (Sparse)
        bm25_results = []
        if self.bm25:
            tokenized_query = self._tokenize(query_text)

            # get_top_n returns raw cache items (dicts with id and metadata)
            top_docs_bm25 = self.bm25.get_top_n(tokenized_query, self.documents_cache, n=top_k)

            for doc in top_docs_bm25:
                bm25_results.append({
                    'id': doc['id'],
                    'metadata': doc['metadata'],
                    'page_content': doc['metadata'].get('text', ''),
                    'score': 0.0,  # BM25Okapi (rank_bm25) doesn't expose score easily here, placeholder for ReRanker
                    'source': 'bm25'
                })

        # C. Fusion and Deduplication (Key: ID)
        final_results = self._fuse_results(vector_results, bm25_results)

        logger.info(
            f"🔎 Hybrid Search: {len(vector_results)} (Vector) + {len(bm25_results)} (BM25) -> {len(final_results)} Unique")

        return final_results

    def _fuse_results(self, vector_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merges results from Vector search and BM25 search based on Document ID.

        Fusion Strategy:
        1. Prioritize Vector results.
        2. Supplement with BM25 results.
        3. Mark duplicates as 'hybrid'.

        Args:
            vector_results (List[Dict[str, Any]]): Results from vector store.
            bm25_results (List[Dict[str, Any]]): Results from BM25.

        Returns:
            List[Dict[str, Any]]: Deduplicated and merged list of documents.
        """
        combined_docs: Dict[str, Dict[str, Any]] = {}

        # 1. Priority to Vector
        for doc in vector_results:
            doc_id = doc.get('id')
            if doc_id:
                combined_docs[doc_id] = doc

        # 2. Complement with BM25
        for doc in bm25_results:
            doc_id = doc.get('id')
            if doc_id:
                if doc_id not in combined_docs:
                    combined_docs[doc_id] = doc
                else:
                    # If it already exists, mark as hybrid (found by both methods)
                    combined_docs[doc_id]['source'] = 'hybrid'

        return list(combined_docs.values())
