class RAGBaseError(Exception):
    """Base exception class for all RAG system errors."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception

class VectorStoreError(RAGBaseError):
    """Raised when an operation on the Vector Store fails."""
    pass

class EmbeddingError(RAGBaseError):
    """Raised when embedding generation fails."""
    pass

class IngestionError(RAGBaseError):
    """Raised when document ingestion or processing fails."""
    pass

class LLMError(RAGBaseError):
    """Raised when LLM generation or initialization fails."""
    pass
