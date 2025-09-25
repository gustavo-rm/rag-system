from typing import Dict, Any
from .base import VectorStore
from .pinecone_store import PineconeStore
from .chroma_store import ChromaStore

def get_vector_store(config: Dict[str, Any]) -> VectorStore:
    """
    Fábrica que cria e retorna uma instância do VectorStore com base na configuração.

    Exemplo de config:
    config = {
        'type': 'chroma',  # ou 'pinecone'
        'path': './chroma_db_main',
        'collection_name': 'my_app'
    }
    ou
    config = {
        'type': 'pinecone',
        'api_key': 'SUA_API_KEY',
        'environment': 'us-west1-gcp',
        'index_name': 'my-pinecone-index',
        'dimension': 768
    }
    """
    store_type = config.get("type", "chroma").lower()

    if store_type == "pinecone":
        # Valida se as chaves necessárias para o Pinecone existem
        required_keys = ['api_key', 'environment', 'index_name', 'dimension']
        if not all(key in config for key in required_keys):
            raise ValueError(f"Configuração para Pinecone requer as chaves: {required_keys}")

        return PineconeStore(
            api_key=config['api_key'],
            environment=config['environment'],
            index_name=config['index_name'],
            dimension=config['dimension']
        )
    elif store_type == "chroma":
        return ChromaStore(
            path=config.get('path', './chroma_db'),
            collection_name=config.get('collection_name', 'rag_collection')
        )
    else:
        raise ValueError(f"Tipo de Vector Store desconhecido: '{store_type}'. Escolha 'pinecone' ou 'chroma'.")
