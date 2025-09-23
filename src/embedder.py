# /src/embedder.py (Versão Refatorada)

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

# --- Recomendações de Modelos Multilíngues (SBERT) ---
# Modelo Padrão (bom equilíbrio entre performance e velocidade):
DEFAULT_SBERT_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
# Modelo de Alta Performance (pode ser mais lento, mas geralmente mais preciso):
# HIGH_PERFORMANCE_SBERT_MODEL = 'intfloat/multilingual-e5-large'

# --- Modelo Padrão da OpenAI ---
DEFAULT_OPENAI_MODEL = "text-embedding-ada-002"


class Embedder:
    """
    Classe otimizada para gerar embeddings de texto usando SBERT (local) ou OpenAI (API).
    Esta versão utiliza modelos mais adequados para múltiplos idiomas e aplica
    as melhores práticas para eficiência e correção matemática.
    """

    def __init__(self, method: str = 'sbert', model_name: str = None, openai_api_key: str = None):
        """
        Inicializa a classe Embedder.

        Parâmetros:
        - method (str): 'sbert' para modelos locais ou 'openai' para a API.
        - model_name (str): O nome do modelo a ser usado. Se for None, usará um padrão otimizado.
        - openai_api_key (str): Chave da API da OpenAI, necessária se method='openai'.
        """
        self.method = method

        if self.method == 'sbert':
            # Usa o modelo padrão recomendado se nenhum for especificado.
            sbert_model_to_load = model_name or DEFAULT_SBERT_MODEL
            print(f"Carregando modelo SBERT local: {sbert_model_to_load}")
            # device='cuda' pode ser adicionado se você tiver uma GPU NVIDIA configurada.
            self.model = SentenceTransformer(sbert_model_to_load)

        elif self.method == 'openai':
            if not openai_api_key:
                raise ValueError("A chave da API da OpenAI é necessária para usar este método.")
            self.client = OpenAI(api_key=openai_api_key)
            self.openai_model = model_name or DEFAULT_OPENAI_MODEL

        else:
            raise ValueError("Método de embedding inválido. Escolha 'sbert' ou 'openai'.")

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Gera embeddings para uma lista de chunks de texto.

        Parâmetros:
        - chunks (List[str]): Lista de textos para gerar embeddings.

        Retorna:
        - List[List[float]]: Uma lista de vetores de embedding.
        """
        if self.method == 'sbert':
            return self._generate_sbert_embeddings(chunks)
        elif self.method == 'openai':
            return self._generate_openai_embeddings(chunks)

    def _generate_sbert_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Gera embeddings usando Sentence-BERT de forma otimizada.
        """
        # A biblioteca sentence-transformers recomenda usar normalize_embeddings=True
        # para busca por similaridade de cosseno. É mais eficiente do que fazer manualmente.
        embeddings = self.model.encode(
            chunks,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        # O resultado já é um numpy.ndarray, convertemos para lista de listas
        return embeddings.astype(np.float32).tolist()

    def _generate_openai_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Gera embeddings usando a API da OpenAI de forma eficiente (em lote).
        """
        # A API da OpenAI é otimizada para receber uma lista de textos de uma vez.
        # Evita fazer um loop e uma chamada de API para cada chunk.
        response = self.client.embeddings.create(
            input=chunks,
            model=self.openai_model
        )

        # Extrai os embeddings da resposta
        embeddings = [item.embedding for item in response.data]

        # Os embeddings da OpenAI já são normalizados. Não é preciso fazer nada.
        return embeddings
