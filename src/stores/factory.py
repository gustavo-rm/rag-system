from typing import Dict, Any
from .base import VectorStore
from .pinecone_store import PineconeStore
from .chroma_store import ChromaStore

def get_vector_store(config: Dict[str, Any]) -> VectorStore:
    """
    Factory that creates and returns a VectorStore instance based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        VectorStore: The instantiated vector store.

    Raises:
        ValueError: If the store type is unknown or configuration is missing keys.

    Example config:
    config = {
        'type': 'chroma',  # or 'pinecone'
        'path': './chroma_db_main',
        'collection_name': 'my_app'
    }
    or
    config = {
        'type': 'pinecone',
        'api_key': 'YOUR_API_KEY',
        'environment': 'us-west1-gcp',
        'index_name': 'my-pinecone-index',
        'dimension': 768
    }
    """
    store_type = config.get("type", "chroma").lower()

    if store_type == "pinecone":
        # Validates if necessary keys for Pinecone exist
        required_keys = ['api_key', 'environment', 'index_name', 'dimension']
        if not all(key in config for key in required_keys):
            raise ValueError(f"Pinecone configuration requires keys: {required_keys}")

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
        raise ValueError(f"Unknown Vector Store type: '{store_type}'. Choose 'pinecone' or 'chroma'.")
