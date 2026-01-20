import os
import chromadb
from typing import List, Dict, Any, Optional
from .base import VectorStore
import logging

# Configuração de Logger
logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """Implementação do VectorStore para o banco de dados local ChromaDB."""

    def __init__(self, path: str = "./chroma_db", collection_name: str = "rag_collection"):
        # Garante que o diretório de persistência exista
        if not os.path.exists(path):
            os.makedirs(path)

        # Configurações para evitar warnings de telemetria
        settings = chromadb.config.Settings(anonymized_telemetry=False)

        self.client = chromadb.PersistentClient(path=path, settings=settings)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Conectado à coleção '{self.collection_name}' do ChromaDB com sucesso.")

    def store_embeddings(self,
                         chunks: List[str],
                         embeddings: List[List[float]],
                         ids: List[str] = None,
                         metadatas: Optional[List[Dict[str, Any]]] = None):

        if ids is None:
            ids = [str(i) for i in range(len(chunks))]

        # ChromaDB exige que metadatas seja None ou uma lista de Dicts válida.
        # Se metadatas for None, criamos dicts vazios ou deixamos None se a lib aceitar.
        # Para garantir compatibilidade futura (BM25), vamos garantir que existe metadata.
        if metadatas is None:
            metadatas = [{} for _ in range(len(chunks))]

        # Opcional: Salvar o texto dentro do metadata também,
        # embora o Chroma salve em 'documents', isso facilita interoperabilidade.
        for i, meta in enumerate(metadatas):
            meta['text'] = chunks[i]

        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"{len(chunks)} embeddings e metadados armazenados no ChromaDB.")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Busca os chunks de texto mais relevantes.
        Retorna estrutura compatível: {'id', 'score', 'metadata': {'text': ..., 'source': ...}}
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            # Importante: Pedir explicitamente metadatas e documents
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results['ids']:
            # Chroma retorna listas de listas (batch query). Pegamos o índice [0].
            ids_list = results['ids'][0]
            distances_list = results['distances'][0]
            documents_list = results['documents'][0]
            metadatas_list = results['metadatas'][0]

            for i in range(len(ids_list)):

                # Prepara o metadata
                meta = metadatas_list[i] if metadatas_list[i] else {}

                # Garante que o texto esteja acessível via metadata['text']
                # (Necessário para o HybridRetriever funcionar bem)
                if 'text' not in meta:
                    meta['text'] = documents_list[i]

                formatted_results.append({
                    'id': ids_list[i],
                    # Converte distância Cosseno (0 a 2) para Similaridade (1 a -1)
                    # Nota: Chroma retorna distância angular/cosseno. Quanto menor, melhor.
                    'score': 1 - distances_list[i],
                    'metadata': meta
                })

        return formatted_results

    def delete(self):
        logger.info(f"Deletando coleção ChromaDB '{self.collection_name}'...")
        self.client.delete_collection(name=self.collection_name)
        logger.info("Coleção deletada.")