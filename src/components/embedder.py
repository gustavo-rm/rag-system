import torch
import logging
from typing import List
from sentence_transformers import SentenceTransformer

# Logger Configuration
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Default Model Configurations
DEFAULT_SBERT_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


class Embedder:
    """
    Unified high-performance engine for generating Embeddings (Semantic Vectors).

    This class abstracts the interface between local models (via Sentence-Transformers/SBERT)
    and remote APIs (via OpenAI), providing an automatic optimization layer.

    Implementation highlights:
    - **Hardware Acceleration:** Automatic GPU (CUDA) detection to maximize throughput.
    - **Dynamic Batching:** Intelligent batch size management to avoid
      Out Of Memory (OOM) errors on local GPUs and respect 'Rate Limits' on external APIs.
    - **Normalization:** Ensures output vectors are normalized (L2 norm),
      essential for accurate cosine similarity calculations.
    """

    def __init__(self, method: str = 'sbert', model_name: str = None, device: str = None,
                 openai_api_key: str = None, batch_size: int = None):
        """
        Initializes the Embedder with hardware and performance configurations.

        Args:
            method (str): 'sbert' (local) or 'openai' (api).
            model_name (str, optional): Name of the model.
            openai_api_key (str, optional): API Key (only for method='openai').
            batch_size (int, optional): Processing batch size.
                                        If None, it will be automatically defined based on hardware:
                                        - GPU: 32
                                        - CPU: 8
                                        - OpenAI: 100
        """
        self.method = method

        # For 8GB GPUs, the Embedder MUST stay on CPU to leave room for the LLM.
        self.device = self._resolve_device(method) if device is None else device # Forces CPU if not specified

        # --- Logic to define Batch Size ---
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            # Automatic definition of safe defaults
            if self.method == 'openai':
                self.batch_size = 100  # API handles larger batches
            elif self.device == 'cuda':
                self.batch_size = 32  # Safe default for average GPU
            else:
                self.batch_size = 8  # Conservative default for CPU

        logger.info(f"⚙️ Embedder Configuration: Device={self.device.upper()} | Batch Size={self.batch_size}")

        # Model Initialization
        if self.method == 'sbert':
            logger.info(f"🖥️ Initializing SBERT ({model_name or DEFAULT_SBERT_MODEL})...")
            sbert_model = model_name or DEFAULT_SBERT_MODEL
            self.model = SentenceTransformer(
                sbert_model,
                device=self.device,
                tokenizer_kwargs={"fix_mistral_regex": True}
            )

        elif self.method == 'openai':
            if not OpenAI:
                raise ImportError("'openai' library not installed.")
            if not openai_api_key:
                raise ValueError("API Key is required for OpenAI method.")

            self.client = OpenAI(api_key=openai_api_key)
            self.openai_model = model_name or DEFAULT_OPENAI_MODEL
            logger.info(f"☁️ OpenAI Embedder ready: {self.openai_model}")

        else:
            raise ValueError("Invalid method. Use 'sbert' or 'openai'.")

    def _resolve_device(self, method: str, prefer_gpu: bool = False) -> str:
        """
        Resolves the device to be used (CPU or CUDA).

        Args:
            method (str): The embedding method ('sbert' or 'openai').
            prefer_gpu (bool): Whether to prefer GPU if available.

        Returns:
            str: 'cuda' or 'cpu'.
        """
        if not torch.cuda.is_available() and method != 'sbert':
            return "cpu"

        return "cuda" if prefer_gpu else "cpu"

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generates vector representation for a list of texts.

        Args:
            chunks (List[str]): List of text chunks to embed.

        Returns:
            List[List[float]]: List of embeddings.
        """
        if not chunks:
            return []

        if self.method == 'sbert':
            return self._generate_sbert_embeddings(chunks)
        elif self.method == 'openai':
            return self._generate_openai_embeddings(chunks)

    def _generate_sbert_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generates local embeddings using Sentence-Transformers.
        Uses the self.batch_size defined at initialization.

        Args:
            chunks (List[str]): List of text chunks.

        Returns:
            List[List[float]]: List of embeddings.
        """
        embeddings = self.model.encode(
            chunks,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def _generate_openai_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generates embeddings via API with batch handling.
        Uses self.batch_size to control API requests.

        Args:
            chunks (List[str]): List of text chunks.

        Returns:
            List[List[float]]: List of embeddings.
        """
        all_embeddings = []

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i: i + self.batch_size]
            batch = [text.replace("\n", " ") for text in batch]

            try:
                response = self.client.embeddings.create(input=batch, model=self.openai_model)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.info(f"⚠️ Error generating OpenAI embeddings in batch {i}: {e}")
                raise e

        return all_embeddings
