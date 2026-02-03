import os
from pathlib import Path

class Config:
    """
    Centralized configuration for the RAG system.
    Stores model identifiers, paths, database settings, and runtime parameters.
    """

    # --- Paths ---
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    MODELS_DIR = ROOT_DIR / "models"
    LOGS_DIR = ROOT_DIR / "logs"

    # Specific file paths
    DEFAULT_PDF_PATH = DATA_DIR / "pdfs" / "relevo-brasileiro.pdf"
    CHROMA_DB_PATH = DATA_DIR / "chromaDB"
    CHAT_LOGS_DIR = DATA_DIR / "chat_logs"
    EXTRACTED_IMAGES_DIR = "extracted_images" # Kept relative as per original usage or can be absolute

    # --- Models ---
    # LLM
    DEFAULT_LOCAL_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
    TRAINING_LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"

    # Embeddings
    DEFAULT_SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
    FINETUNED_MODEL_PATH = MODELS_DIR / "finetuned_v3"

    # Reranker
    DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

    # --- Database ---
    CHROMA_COLLECTION_NAME = "rag_project_v3"
    DEFAULT_COLLECTION_NAME = "rag_collection" # Fallback/Default in store classes

    # --- Runtime Parameters ---
    # Chunker
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TRAINING_CHUNK_SIZE = 384 # Used in train_embedding.py

    # LLM & Generation
    DEFAULT_CONTEXT_WINDOW = 4000
    DEFAULT_MAX_NEW_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.1

    # Caching
    EXACT_CACHE_CAPACITY = 10000
    SEMANTIC_CACHE_CAPACITY = 5000
    SEMANTIC_CACHE_THRESHOLD = 0.92
    DEFAULT_EMBEDDING_DIM = 768 # Safe default for mpnet-base

    # Retrieval
    RETRIEVAL_TOP_K = 20
    RERANK_TOP_N = 5

    # Training
    TRAINING_BATCH_SIZE = 16
    TRAINING_EPOCHS = 3
    SYNTHETIC_EXAMPLES_COUNT = 50

    # Logging
    LOG_FILENAME = "app_rag.log"

    # --- Internationalization ---
    LANGUAGE = "pt-BR" # Default language ("en-US" or "pt-BR")
