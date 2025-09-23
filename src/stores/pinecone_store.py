import pinecone
from typing import List, Dict, Any
from .base import VectorStore

class PineconeStore(VectorStore):
    """Implementação do VectorStore para o serviço em nuvem Pinecone."""

    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int):
        self.pinecone = pinecone.Pinecone(api_key=api_key)
        self.index_name = index_name

        if self.index_name not in self.pinecone.list_indexes().names():
            print(f"Índice Pinecone '{self.index_name}' não encontrado. Criando um novo...")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',
                spec=pinecone.ServerlessSpec(cloud='aws', region=environment)
            )
        self.index = self.pinecone.Index(self.index_name)
        print("Conectado ao índice Pinecone com sucesso.")

    def store_embeddings(self, chunks: List[str], embeddings: List[List[float]], ids: List[str] = None):
        if ids is None:
            ids = [str(i) for i in range(len(chunks))]

        vectors_to_upsert = [
            {'id': ids[i], 'values': embedding, 'metadata': {'text': chunk}}
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        self.index.upsert(vectors=vectors_to_upsert, batch_size=100)
        print(f"{len(vectors_to_upsert)} embeddings armazenados no Pinecone.")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        result = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return result.get('matches', [])

    def delete(self):
        print(f"Deletando índice Pinecone '{self.index_name}'...")
        self.pinecone.delete_index(self.index_name)
        print("Índice deletado.")
