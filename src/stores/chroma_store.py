import os
import chromadb
from typing import List, Dict, Any
from .base import VectorStore

class ChromaStore(VectorStore):
    """Implementação do VectorStore para o banco de dados local ChromaDB."""

    def __init__(self, path: str = "./chroma_db", collection_name: str = "rag_collection"):
        # Garante que o diretório de persistência exista
        if not os.path.exists(path):
            os.makedirs(path)

        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Conectado à coleção '{self.collection_name}' do ChromaDB com sucesso.")

    def store_embeddings(self, chunks: List[str], embeddings: List[List[float]], ids: List[str] = None):
        if ids is None:
            ids = [str(i) for i in range(len(chunks))]

        self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)
        print(f"{len(chunks)} embeddings armazenados no ChromaDB.")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Estrutura o resultado para ser similar ao do Pinecone e mais fácil de usar
        formatted_results = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'score': 1 - results['distances'][0][i],  # Converte distância para similaridade
                    'metadata': {'text': results['documents'][0][i]}
                })
        return formatted_results

    def delete(self):
        print(f"Deletando coleção ChromaDB '{self.collection_name}'...")
        self.client.delete_collection(name=self.collection_name)
        print("Coleção deletada.")
