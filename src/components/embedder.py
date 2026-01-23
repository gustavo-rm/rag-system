import torch
import logging
from typing import List
from sentence_transformers import SentenceTransformer

# Configuração de Logger
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Configurações de Modelos Padrão
DEFAULT_SBERT_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


class Embedder:
    """
    Motor unificado para geração de Embeddings (Vetores Semânticos) de alta performance.

    Esta classe abstrai a interface entre modelos locais (via Sentence-Transformers/SBERT)
    e APIs remotas (via OpenAI), oferecendo uma camada de otimização automática.

    Destaques da implementação:
    - **Aceleração de Hardware:** Detecção automática de GPU (CUDA) para maximizar o throughput.
    - **Batching Dinâmico:** Gerenciamento inteligente do tamanho dos lotes para evitar
      estouro de memória (OOM) em GPUs locais e respeitar 'Rate Limits' em APIs externas.
    - **Normalização:** Garante que os vetores de saída sejam normalizados (norma L2),
      essencial para cálculos precisos de similaridade de cosseno.
    """

    def __init__(self, method: str = 'sbert', model_name: str = None, device: str = None,
                 openai_api_key: str = None, batch_size: int = None):
        """
        Inicializa o Embedder com configurações de hardware e performance.

        Args:
            method (str): 'sbert' (local) ou 'openai' (api).
            model_name (str, optional): Nome do modelo.
            openai_api_key (str, optional): Chave da API (apenas para method='openai').
            batch_size (int, optional): Tamanho do lote de processamento.
                                        Se None, será definido automaticamente com base no hardware:
                                        - GPU: 32
                                        - CPU: 8
                                        - OpenAI: 100
        """
        self.method = method

        # Para GPUs de 8GB, o Embedder DEVE ficar na CPU para deixar espaço pro LLM.
        self.device = self._resolve_device(method) if device is None else device # Força CPU se não especificado

        # --- Lógica para definir o Batch Size ---
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            # Definição automática de defaults seguros
            if self.method == 'openai':
                self.batch_size = 100  # API aguenta lotes maiores
            elif self.device == 'cuda':
                self.batch_size = 32  # Padrão seguro para GPU média
            else:
                self.batch_size = 8  # Padrão conservador para CPU

        logger.info(f"⚙️ Configuração Embedder: Device={self.device.upper()} | Batch Size={self.batch_size}")

        # Inicialização dos Modelos
        if self.method == 'sbert':
            logger.info(f"🖥️ Inicializando SBERT ({model_name or DEFAULT_SBERT_MODEL})...")
            sbert_model = model_name or DEFAULT_SBERT_MODEL
            self.model = SentenceTransformer(
                sbert_model,
                device=self.device,
                tokenizer_kwargs={"fix_mistral_regex": True}
            )

        elif self.method == 'openai':
            if not OpenAI:
                raise ImportError("Biblioteca 'openai' não instalada.")
            if not openai_api_key:
                raise ValueError("API Key é obrigatória para o método OpenAI.")

            self.client = OpenAI(api_key=openai_api_key)
            self.openai_model = model_name or DEFAULT_OPENAI_MODEL
            logger.info(f"☁️ Embedder OpenAI pronto: {self.openai_model}")

        else:
            raise ValueError("Método inválido. Use 'sbert' ou 'openai'.")

    def _resolve_device(self, method: str, prefer_gpu: bool = False) -> str:
        if not torch.cuda.is_available() and method != 'sbert':
            return "cpu"

        return "cuda" if prefer_gpu else "cpu"

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Gera a representação vetorial para uma lista de textos."""
        if not chunks:
            return []

        if self.method == 'sbert':
            return self._generate_sbert_embeddings(chunks)
        elif self.method == 'openai':
            return self._generate_openai_embeddings(chunks)

    def _generate_sbert_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Gera embeddings locais usando Sentence-Transformers.
        Usa o self.batch_size definido na inicialização.
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
        Gera embeddings via API com tratamento de lotes.
        Usa o self.batch_size para controlar requisições à API.
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
                logger.info(f"⚠️ Erro ao gerar embeddings OpenAI no lote {i}: {e}")
                raise e

        return all_embeddings