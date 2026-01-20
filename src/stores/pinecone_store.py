import pinecone
from typing import List, Dict, Any, Optional
from .base import VectorStore
import logging

logger = logging.getLogger(__name__)


class PineconeStore(VectorStore):
    """Implementação do VectorStore para o serviço em nuvem Pinecone."""

    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int):
        # Inicialização do cliente (ajuste conforme versão da lib pinecone-client)
        self.pinecone = pinecone.Pinecone(api_key=api_key)
        self.index_name = index_name

        existing_indexes = [i.name for i in self.pinecone.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(f"Índice Pinecone '{self.index_name}' não encontrado. Criando um novo...")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',
                spec=pinecone.ServerlessSpec(cloud='aws', region=environment)
            )

        self.index = self.pinecone.Index(self.index_name)
        logger.info("Conectado ao índice Pinecone com sucesso.")

    def store_embeddings(self,
                         chunks: List[str],
                         embeddings: List[List[float]],
                         ids: List[str] = None,
                         metadatas: Optional[List[Dict[str, Any]]] = None):

        if ids is None:
            ids = [str(i) for i in range(len(chunks))]

        vectors_to_upsert = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # 1. Pega metadata externo ou cria vazio
            meta = metadatas[i].copy() if metadatas and i < len(metadatas) else {}

            # 2. Insere OBRIGATORIAMENTE o texto no metadata (Pinecone requirement)
            meta['text'] = chunk

            vectors_to_upsert.append({
                'id': ids[i],
                'values': embedding,
                'metadata': meta
            })

        # Upsert em batches para evitar payload too large
        self.index.upsert(vectors=vectors_to_upsert, batch_size=100)
        logger.info(f"{len(vectors_to_upsert)} embeddings armazenados no Pinecone.")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        result = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return result.get('matches', [])

    def delete(self):
        logger.info(f"Deletando índice Pinecone '{self.index_name}'...")
        self.pinecone.delete_index(self.index_name)
        logger.info("Índice deletado.")