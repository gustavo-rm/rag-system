import os
from pathlib import Path
import logging
from dotenv import load_dotenv
from src.utils.logger import setup_logging

# --- Logging Configuration ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Load Variables ---
load_dotenv()

# CRITICAL MEMORY OPTIMIZATION
# Helps avoid OOM (Out of Memory) errors when memory is highly fragmented.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- AUTO-CONFIGURATION OF OFFLINE/ONLINE MODE ---
def auto_configure_huggingface():
    """
    Checks if the necessary models are already in the cache.

    If ALL models are present:
        Activates OFFLINE mode (fast initialization, no timeout).
    If ANY model is missing:
        Activates ONLINE mode (allows downloading).

    Side Effects:
        - Checks the HuggingFace cache directory.
        - Sets 'HF_HUB_OFFLINE' and 'TRANSFORMERS_OFFLINE' environment variables if models are found.
        - Deletes these environment variables if downloads are needed.
    """
    # 1. Definition of models used in the project
    required_models = [
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",  # Current LLM
        "BAAI/bge-reranker-base",  # Reranker
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # Base Embedder
    ]

    # Default cache path on Linux
    cache_root = Path(os.getenv("HF_HOME", "~/.cache/huggingface/hub")).expanduser()

    missing_models = []

    logger.info(f"🔍 Checking cache at: {cache_root}")

    for model_id in required_models:
        # HuggingFace saves folders replacing '/' with '--'
        # Example: unsloth/llama-3 -> models--unsloth--llama-3
        folder_name = f"models--{model_id.replace('/', '--')}"
        model_path = cache_root / folder_name

        if not model_path.exists():
            missing_models.append(model_id)

    if not missing_models:
        logger.info("✅ All models found in local cache.")
        logger.info("🚀 Activating OFFLINE mode (Instant Initialization).")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        logger.info(f"🌐 Missing models detected: {missing_models}")
        logger.info("⬇️  ONLINE mode activated for download.")
        # Ensure variables are not set
        if "HF_HUB_OFFLINE" in os.environ: del os.environ["HF_HUB_OFFLINE"]
        if "TRANSFORMERS_OFFLINE" in os.environ: del os.environ["TRANSFORMERS_OFFLINE"]


# Executes verification before loading the rest of the system
auto_configure_huggingface()

# --- Imports ---

# Ingestion
from src.ingestion.chunker import Chunker

# Core AI Components
from src.components.embedder import Embedder
from src.components.reranker import ReRanker
from src.components.llm import LLM, LLMGenerationError
from src.components.hybrid_retriever import HybridRetriever
from src.routing.query_router import QueryRouter

# Query Transformers
from src.query_transformers import NoOpTransformer, MultiQueryTransformer, HyDETransformer

# Caching & Preprocessing
from src.caching.cache_manager import CacheManager
from src.caching.semantic_cache import SemanticCache
from src.preprocessing.query_corrector import QueryCorrector

# Storage
from src.stores import get_vector_store

# Application Logic
from src.pipeline import RAGSystem
from src.chat.chatbot import Chatbot


def main():
    """
    Main orchestration function for the RAG system.

    This function initializes all components of the Retrieval-Augmented Generation (RAG) pipeline,
    including the vector store, retriever, embedder, LLM, reranker, query router, and cache layers.
    It also handles the ingestion of a default PDF document if it has not been processed yet.

    Finally, it enters an interactive chat loop where the user can ask questions to the system.
    """
    logger.info("🚀 Initializing RAG system...")

    # ==========================================
    # 1. BASE COMPONENTS CONFIGURATION
    # ==========================================

    # --- A. Vector Store (Database) ---
    logger.info("Configuring Vector Store...")
    config_store = {
        'type': 'chroma',
        'path': 'data/chromaDB/',
        'collection_name': 'rag_project_v3'
    }
    base_vector_store = get_vector_store(config_store)

    # --- B. Retriever (Hybrid) ---
    # Wraps the vector store to add keyword search capability (BM25)
    hybrid_retriever = HybridRetriever(base_vector_store)

    # --- C. AI Components (Embedder, LLM, ReRanker) ---

    # Chunker
    chunker = Chunker(chunk_size=512, chunk_overlap=50)

    # --- Embedding Model Configuration ---

    # Path where the training script saved the model
    finetuned_model_path = "models/finetuned_v3"
    base_model_name = "paraphrase-multilingual-mpnet-base-v2"

    # Intelligent selection logic
    if os.path.exists(finetuned_model_path):
        logger.info(f"💎 Fine-Tuned model detected! Using: {finetuned_model_path}")
        selected_model = finetuned_model_path
    else:
        logger.warning(
            f"⚠️ Fine-Tuned model not found in '{finetuned_model_path}'. Using base model: {base_model_name}")
        selected_model = base_model_name

    # LLM: Configured with Context Window control and No-Grad
    # If using OpenAI, remember to configure the key in .env
    llm = LLM(
        method='local',
        model_name='unsloth/llama-3-8b-Instruct-bnb-4bit'
    )

    # Instantiate the Embedder with the chosen model
    embedder = Embedder(
        method='sbert',
        model_name=selected_model
        # batch_size will be auto-configured (32 for GPU)
    )

    # ReRanker: Updated to BAAI model (Better Multi-lingual/PT-BR support)
    reranker = ReRanker(model_name='BAAI/bge-reranker-base', device='cpu')

    # ==========================================
    # 2. ROUTING STRATEGIES (ROUTER)
    # ==========================================
    logger.info("Configuring Query Routing strategies...")

    # Instantiate strategies injecting the LLM where necessary
    transformers_map = {
        "noop": NoOpTransformer(),
        "hyde": HyDETransformer(llm),
        "multi_query": MultiQueryTransformer(llm, num_queries=3)
    }

    # The Router receives the map and will decide which one to use at runtime
    query_router = QueryRouter(llm, strategies=transformers_map)

    # ==========================================
    # 3. CACHE AND PREPROCESSING
    # ==========================================

    # Get dimension dynamically from the loaded model (e.g., 768 for mpnet)
    # Access the internal attribute of SentenceTransformer if the method is sbert
    embedding_dim = 768  # Safe default value for mpnet-base
    if hasattr(embedder, 'model') and hasattr(embedder.model, 'get_sentence_embedding_dimension'):
        embedding_dim = embedder.model.get_sentence_embedding_dimension()

    exact_cache = CacheManager()
    semantic_cache = SemanticCache(dimension=embedding_dim, similarity_threshold=0.92)
    query_corrector = QueryCorrector(language='pt', enable_grammar=True)

    # ==========================================
    # 4. SYSTEM ASSEMBLY (RAG + CHATBOT)
    # ==========================================

    logger.info("Assembling RAG Pipeline...")

    rag_system = RAGSystem(
        chunker=chunker,
        embedder=embedder,
        retriever=hybrid_retriever,
        reranker=reranker,
        llm=llm,
        router=query_router
    )

    chatbot = Chatbot(
        llm=llm,
        rag_system=rag_system,
        cache_manager=exact_cache,
        semantic_cache=semantic_cache,
        query_corrector=query_corrector
    )

    # ==========================================
    # 5. DATA INGESTION (Single Execution)
    # ==========================================

    pdf_path = "data/pdfs/relevo-brasileiro.pdf"

    # Checks if the file exists
    if os.path.exists(pdf_path):
        # Simple logic to avoid re-ingestion on every boot
        # In production, check if the file is already in the database by hash or name
        ingestion_done_marker = f"{pdf_path}.done"

        if not os.path.exists(ingestion_done_marker):
            logger.info(f"Starting ingestion of document: {pdf_path}")
            try:
                rag_system.setup_pipeline(pdf_path)
                # Create an empty file to mark as done
                with open(ingestion_done_marker, 'w') as f:
                    f.write('done')
                logger.info("Ingestion completed and marked.")
            except Exception as e:
                logger.error(f"Ingestion failed: {e}")
        else:
            logger.info("Document previously processed. Skipping ingestion.")
    else:
        logger.warning(f"PDF not found at '{pdf_path}'. The system will work only with prior knowledge.")

    # ==========================================
    # 6. INTERACTION LOOP (CHAT)
    # ==========================================

    print("\n" + "=" * 50)
    print("🤖 RAG Assistant v3.0 Ready!")
    print("Commands: 'exit' to quit.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_question = input("You: ").strip()

            if not user_question:
                continue

            if user_question.lower() in ['sair', 'exit', 'quit']:
                logger.info("Ending session.")
                print("Assistant: See you later! 👋")
                break

            # The Chatbot manages the entire flow (correction -> cache -> RAG -> Response)
            response = chatbot.chat(user_question)

            print(f"Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except LLMGenerationError as e:
            logger.error(f"LLM Error: {e}")
            print("Assistant: Sorry, I had a problem generating the response. Try simplifying the question.")
        except Exception as e:
            logger.critical(f"Unhandled error: {e}")
            print("Assistant: An internal error occurred.")


if __name__ == "__main__":
    main()
