from openai import OpenAI
from sentence_transformers import SentenceTransformer


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

    def _generate_sbert_embeddings(self, chunks):
        embeddings = self.model.encode(chunks)
        return embeddings

    def _generate_openai_embeddings(self, chunks):
        embeddings = []
        for chunk in chunks:
            response = self.client.embeddings.create(input=chunk, model="text-embedding-ada-002")
            embeddings.append(response.data[0].embedding)
        return embeddings
