from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, method='sbert', openai_api_key=None):
        """
        method: 'sbert' ou 'openai'
        openai_api_key: chave da API da OpenAI, necessária se method='openai'
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
        if self.method == 'sbert':
            return self._generate_sbert_embeddings(chunks)
        elif self.method == 'openai':
            return self._generate_openai_embeddings(chunks)

    def validate_and_normalize_embedding(self, embedding):
        """
        Valida e normaliza um vetor de embedding.
        - Substitui valores NaN e infinitos por 0.
        - Normaliza o vetor para garantir magnitude 1.
        - Limita os valores entre 0 e 1 (ao invés de -1 e 1).
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
        embeddings = self.model.encode(chunks)

        # Validar e normalizar os embeddings
        embeddings = [self.validate_and_normalize_embedding(embedding) for embedding in embeddings]

        return embeddings

    def _generate_openai_embeddings(self, chunks):
        embeddings = []
        for chunk in chunks:
            response = self.client.embeddings.create(input=chunk, model="text-embedding-ada-002")
            embedding = response['data'][0]['embedding']

            # Validar e normalizar o embedding
            embedding = self.validate_and_normalize_embedding(np.array(embedding))

            embeddings.append(embedding.tolist())  # Converter para lista quando necessário

        return embeddings
