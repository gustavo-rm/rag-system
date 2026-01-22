import os
import chromadb
from typing import List, Dict, Any, Optional
from .base import VectorStore
import logging

# Logger Configuration
logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """VectorStore implementation for the local database ChromaDB."""

    def __init__(self, path: str = "./chroma_db", collection_name: str = "rag_collection"):
        """
        Initializes the ChromaDB store.

        Args:
            path (str): Path to the persistence directory.
            collection_name (str): Name of the collection to use.
        """
        # Ensures the persistence directory exists
        if not os.path.exists(path):
            os.makedirs(path)

        # Settings to avoid telemetry warnings
        settings = chromadb.config.Settings(anonymized_telemetry=False)

        self.client = chromadb.PersistentClient(path=path, settings=settings)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Successfully connected to ChromaDB collection '{self.collection_name}'.")

    def store_embeddings(self,
                         chunks: List[str],
                         embeddings: List[List[float]],
                         ids: List[str] = None,
                         metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Stores embeddings and associated data in ChromaDB.

        Args:
            chunks (List[str]): List of texts.
            embeddings (List[List[float]]): List of vectors.
            ids (List[str], optional): List of unique IDs.
            metadatas (Optional[List[Dict[str, Any]]]): Metadata for each chunk.
        """
        if ids is None:
            ids = [str(i) for i in range(len(chunks))]

        # ChromaDB requires metadatas to be None or a valid list of Dicts.
        # To ensure future compatibility (e.g., BM25), we ensure metadata exists.
        if metadatas is None:
            metadatas = [{} for _ in range(len(chunks))]

        # Optional: Save text inside metadata as well,
        # although Chroma saves it in 'documents', this facilitates interoperability.
        for i, meta in enumerate(metadatas):
            meta['text'] = chunks[i]

        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"{len(chunks)} embeddings and metadata stored in ChromaDB.")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for the most relevant text chunks.

        Args:
            query_embedding (List[float]): The query vector.
            top_k (int): Number of results to return.

        Returns:
            List[Dict[str, Any]]: Compatible structure: {'id', 'score', 'metadata': {'text': ..., 'source': ...}}
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            # Important: Explicitly request metadatas and documents
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results['ids']:
            # Chroma returns lists of lists (batch query). We take index [0].
            ids_list = results['ids'][0]
            distances_list = results['distances'][0]
            documents_list = results['documents'][0]
            metadatas_list = results['metadatas'][0]

            for i in range(len(ids_list)):

                # Prepare metadata
                meta = metadatas_list[i] if metadatas_list[i] else {}

                # Ensure text is accessible via metadata['text']
                # (Required for HybridRetriever to work well)
                if 'text' not in meta:
                    meta['text'] = documents_list[i]

                formatted_results.append({
                    'id': ids_list[i],
                    # Convert Cosine distance (0 to 2) to Similarity (1 to -1)
                    # Note: Chroma returns angular/cosine distance. Smaller is better.
                    'score': 1 - distances_list[i],
                    'metadata': meta
                })

        return formatted_results

    def delete(self):
        """Deletes the ChromaDB collection."""
        logger.info(f"Deleting ChromaDB collection '{self.collection_name}'...")
        self.client.delete_collection(name=self.collection_name)
        logger.info("Collection deleted.")
