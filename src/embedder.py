from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, method='sbert', openai_api_key=None):
        """
        Inicializa a classe Embedder para gerar embeddings de chunks de texto.

        Parâmetros:
        - method: Método de geração de embeddings. Pode ser 'sbert' (Sentence-BERT) ou 'openai' (modelos da OpenAI).
        - openai_api_key: Chave da API da OpenAI, necessária se o método 'openai' for utilizado.

        O modelo 'all-MiniLM-L6-v2' é utilizado no caso do método 'sbert'. Se o método for 'openai', a chave da API
        da OpenAI é necessária para acessar os modelos de embeddings.
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.method = method
        if method == 'sbert':
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        elif method == 'openai':
            if openai_api_key is None:
                raise ValueError("Chave da API da OpenAI é necessária para usar OpenAI embeddings.")
        else:
            raise ValueError("Método de embedding inválido.")

    def generate_embeddings(self, chunks):
        """
        Gera embeddings para uma lista de chunks de texto, de acordo com o método especificado.

        Parâmetros:
        - chunks: Lista de pedaços (chunks) de texto para os quais os embeddings serão gerados.

        Retorna:
        - Uma lista de embeddings gerados, que podem ser gerados via 'sbert' ou 'openai'.
        """
        if self.method == 'sbert':
            return self._generate_sbert_embeddings(chunks)
        elif self.method == 'openai':
            return self._generate_openai_embeddings(chunks)

    def validate_and_normalize_embedding(self, embedding):
        """
        Valida e normaliza um vetor de embedding.

        Parâmetros:
        - embedding: O vetor de embedding a ser normalizado.

        Processos realizados:
        - Substitui valores NaN e infinitos por 0.
        - Normaliza o vetor para garantir que sua magnitude seja 1.
        - Limita os valores do embedding entre 0.001 e 1.0, para evitar valores absolutos de 0.
        - Garante que os valores do vetor sejam do tipo float32.

        Retorna:
        - O vetor de embedding validado e normalizado.
        """
        # Substituir NaN, infinitos positivos e negativos por 0
        embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalizar o vetor (magnitude do vetor = 1)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Limitar os valores entre 0.001 (Evitar valores de 0 absolutos) e 1
        embedding = np.clip(embedding, 0.001, 1.0)

        # Garantir que o embedding está em formato de float32
        embedding = embedding.astype(np.float32)

        return embedding

    def _generate_sbert_embeddings(self, chunks):
        """
        Gera embeddings usando Sentence-BERT (SBERT) para uma lista de chunks de texto.

        Parâmetros:
        - chunks: Lista de pedaços (chunks) de texto para os quais os embeddings serão gerados.

        Retorna:
        - Uma lista de embeddings SBERT normalizados.
        """
        embeddings = self.model.encode(chunks)

        # Validar e normalizar os embeddings
        embeddings = [self.validate_and_normalize_embedding(embedding) for embedding in embeddings]

        return embeddings

    def _generate_openai_embeddings(self, chunks):
        """
        Gera embeddings usando a API da OpenAI para uma lista de chunks de texto.

        Parâmetros:
        - chunks: Lista de pedaços (chunks) de texto para os quais os embeddings serão gerados.

        Retorna:
        - Uma lista de embeddings gerados pela OpenAI, que são validados e normalizados.
        """
        embeddings = []
        for chunk in chunks:
            response = self.client.embeddings.create(input=chunk, model="text-embedding-ada-002")
            embedding = response['data'][0]['embedding']

            # Validar e normalizar o embedding
            embedding = self.validate_and_normalize_embedding(np.array(embedding))

            embeddings.append(embedding.tolist())  # Converter para lista quando necessário

        return embeddings
