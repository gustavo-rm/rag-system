from src.extractor import PDFExtractor
from src.chunker import Chunker
from src.embedder import Embedder
from src.embedding_store import EmbeddingStore
from src.llm import LLM
from dotenv import load_dotenv
import os

# Carregar as variáveis de ambiente do .env
load_dotenv()

var_pinecone_api_key = os.getenv("PINECONE_API_KEY")
var_pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
var_openai_api_key = os.getenv("OPENAI_API_KEY")


class RAGSystem:
    def __init__(self,
                 pdf_path,
                 chunk_method='sentences',
                 chunk_size=100,
                 embedder_method='sbert',
                 openai_api_key=None,
                 pinecone_api_key=None,
                 pinecone_environment=None,
                 embedding_dimension=384,
                 index_name="my-vector-index",
                 llm_method='openai',
                 local_llm_model_name="EleutherAI/gpt-neo-2.7B"):
        # Extração
        self.extractor = PDFExtractor(pdf_path)

        # Chunking
        self.chunker = Chunker(method=chunk_method, chunk_size=chunk_size)

        # Embedding
        self.embedder = Embedder(method=embedder_method, openai_api_key=openai_api_key)

        # Embedding Store (Pinecone)
        self.embedding_store = EmbeddingStore(
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_environment,
            dimension=embedding_dimension,
            index_name=index_name
        )

        # LLM
        self.llm = LLM(method=llm_method, openai_api_key=openai_api_key, local_model_name=local_llm_model_name)

        # Inicializar chunks armazenados
        self.chunks = []

    def prepare_data(self):
        # Extrair texto do PDF
        text = self.extractor.extract_text()

        # Dividir o texto em chunks
        self.chunks = self.chunker.chunk_text(text)
        print(f"{len(self.chunks)} chunks criados.")

        # Gerar embeddings para cada chunk
        embeddings = self.embedder.generate_embeddings(self.chunks)

        # Gerar IDs para cada embedding (IDs podem ser os índices dos chunks)
        ids = [str(i) for i in range(len(embeddings))]

        # Armazenar embeddings no Pinecone
        self.embedding_store.store_embeddings(embeddings, ids=ids)

    def query(self, user_query, top_k=5):
        # Gerar embedding para a consulta do usuário
        query_embedding = self.embedder.generate_embeddings([user_query])[0]

        # Buscar os chunks mais relevantes usando o embedding da consulta
        matches = self.embedding_store.search(query_embedding, top_k=top_k)

        # Recuperar os chunks relevantes com base nos IDs retornados
        relevant_chunks = [self.chunks[int(match['id'])] for match in matches]

        # Concatenar os chunks para formar o contexto
        context = "\n".join(relevant_chunks)

        # Criar o prompt para o LLM
        prompt = f"Aqui estão algumas informações relevantes extraídas de documentos:\n{context}\n\nCom base nessas informações, responda à seguinte pergunta:\n{user_query}"

        # Gerar resposta com o LLM
        answer = self.llm.generate_response(prompt)

        return answer, relevant_chunks
