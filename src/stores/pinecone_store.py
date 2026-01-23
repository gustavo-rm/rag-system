import pinecone
from typing import List, Dict, Any, Optional
from .base import VectorStore
import logging

logger = logging.getLogger(__name__)


class PineconeStore(VectorStore):
    """VectorStore implementation for the Pinecone cloud service."""

    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int):
        """
        Initializes the Pinecone store.

        Args:
            api_key (str): Pinecone API Key.
            environment (str): Pinecone environment (region).
            index_name (str): Name of the index.
            dimension (int): Dimension of vectors to be stored.
        """
        # Client initialization (adjust according to pinecone-client lib version)
        self.pinecone = pinecone.Pinecone(api_key=api_key)
        self.index_name = index_name

        existing_indexes = [i.name for i in self.pinecone.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(f"Pinecone index '{self.index_name}' not found. Creating a new one...")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',
                spec=pinecone.ServerlessSpec(cloud='aws', region=environment)
            )

        self.index = self.pinecone.Index(self.index_name)
        logger.info("Successfully connected to Pinecone index.")

    def store_embeddings(self,
                         chunks: List[str],
                         embeddings: List[List[float]],
                         ids: List[str] = None,
                         metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Stores embeddings and associated data in Pinecone.

        Args:
            chunks (List[str]): List of texts.
            embeddings (List[List[float]]): List of vectors.
            ids (List[str], optional): List of unique IDs.
            metadatas (Optional[List[Dict[str, Any]]]): Metadata for each chunk.
        """
        if ids is None:
            ids = [str(i) for i in range(len(chunks))]

        vectors_to_upsert = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # 1. Get external metadata or create empty
            meta = metadatas[i].copy() if metadatas and i < len(metadatas) else {}

            # 2. MANDATORY: Insert text into metadata (Pinecone requirement)
            meta['text'] = chunk

            vectors_to_upsert.append({
                'id': ids[i],
                'values': embedding,
                'metadata': meta
            })

        # Upsert in batches to avoid payload too large
        self.index.upsert(vectors=vectors_to_upsert, batch_size=100)
        logger.info(f"{len(vectors_to_upsert)} embeddings stored in Pinecone.")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for the most relevant vectors in Pinecone.

        Args:
            query_embedding (List[float]): The query vector.
            top_k (int): Number of results to return.

        Returns:
            List[Dict[str, Any]]: List of matches.
        """
        result = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return result.get('matches', [])

    def delete(self):
        """Deletes the Pinecone index."""
        logger.info(f"Deleting Pinecone index '{self.index_name}'...")
        self.pinecone.delete_index(self.index_name)
        logger.info("Index deleted.")
