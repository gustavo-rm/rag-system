from src.evaluator import Evaluator
from src.extractor import PDFExtractor
from src.chunker import Chunker
from src.embedder import Embedder
from src.embedding_store import EmbeddingStore
from src.llm import LLM
import numpy as np


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
                 local_llm_model_name="EleutherAI/gpt-neo-2.7B",
                 evaluator=None):
        """
        Inicializa o sistema RAG (Retrieval-Augmented Generation), que combina a extração de dados de um PDF,
        a divisão do texto em chunks, a criação de embeddings e a geração de respostas com um LLM.

        Parâmetros:
        - pdf_path: Caminho para o arquivo PDF que será extraído.
        - chunk_method: Método de chunking ('sentences', 'paragraphs', 'tokens') para dividir o texto extraído.
        - chunk_size: Tamanho máximo de cada chunk em caracteres ou tokens.
        - embedder_method: Método de embedding ('sbert' ou 'openai') para gerar vetores de embeddings.
        - openai_api_key: Chave da API OpenAI, necessária se embedder_method ou llm_method for 'openai'.
        - pinecone_api_key: Chave da API do Pinecone, usada para armazenar e consultar os embeddings.
        - pinecone_environment: Ambiente do Pinecone (ex: 'us-west1-gcp').
        - embedding_dimension: Dimensão dos embeddings gerados.
        - index_name: Nome do índice do Pinecone para armazenar embeddings.
        - llm_method: Método para gerar respostas, 'openai' ou 'local' (modelo Hugging Face).
        - local_llm_model_name: Nome do modelo local para geração de texto (se llm_method for 'local').
        - evaluator: Objeto de avaliação de respostas (usando métricas como BLEU ou ROUGE), opcional.
        """
        # Extração de texto do PDF
        self.extractor = PDFExtractor(pdf_path)

        # Divisão do texto em chunks
        self.chunker = Chunker(method=chunk_method, chunk_size=chunk_size)

        # Criação de embeddings para os chunks
        self.embedder = Embedder(method=embedder_method, openai_api_key=openai_api_key)

        # Armazenamento de embeddings no Pinecone
        self.embedding_store = EmbeddingStore(
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_environment,
            dimension=embedding_dimension,
            index_name=index_name
        )

        # Inicializa o LLM (Language Model) para gerar respostas
        self.llm = LLM(method=llm_method, openai_api_key=openai_api_key, local_model_name=local_llm_model_name)

        # Inicializa o avaliador para calcular métricas (se não for fornecido, cria uma instância)
        self.evaluator = evaluator if evaluator else Evaluator()

        # Inicializa a lista de chunks armazenados
        self.chunks = []

    def prepare_data(self):
        """
        Prepara os dados do sistema RAG, extraindo texto do PDF, dividindo-o em chunks, gerando embeddings e
        armazenando-os no Pinecone.

        Passos:
        1. Extrai o texto do arquivo PDF.
        2. Divide o texto em chunks de acordo com o método escolhido (sentences, paragraphs, tokens).
        3. Gera embeddings para cada chunk de texto.
        4. Armazena os embeddings no Pinecone, associando cada embedding a um ID exclusivo.
        """
        # Extrair texto do PDF
        text = self.extractor.extract_text()

        # Dividir o texto em chunks
        self.chunks = self.chunker.chunk_text(text)
        print(f"{len(self.chunks)} chunks criados.")

        # Gerar embeddings para cada chunk
        embeddings = self.embedder.generate_embeddings(self.chunks)

        # Gerar IDs para os embeddings (usando o índice dos chunks)
        ids = [str(i) for i in range(len(embeddings))]

        # Armazenar os embeddings no Pinecone
        self.embedding_store.store_embeddings(embeddings, ids=ids)

    def query(self, user_query, reference_answer=None, top_k=5):
        """
        Faz uma consulta ao sistema RAG, utilizando embeddings e um modelo de linguagem para responder à pergunta do usuário.

        Parâmetros:
        - user_query: A pergunta ou consulta do usuário.
        - reference_answer: Resposta de referência para avaliação (opcional).
        - top_k: Número de chunks mais relevantes a serem retornados na busca.

        Retorna:
        - answer: A resposta gerada pelo modelo LLM.
        - relevant_chunks: Os chunks mais relevantes encontrados para a consulta.
        """
        # Gerar embedding para a consulta do usuário
        query_embedding = self.embedder.generate_embeddings([user_query])[0]

        # Validar e normalizar o embedding gerado
        query_embedding = self.embedder.validate_and_normalize_embedding(np.array(query_embedding))

        # Limitar o número de casas decimais dos valores para 6
        query_embedding = [round(float(x), 9) for x in query_embedding]

        # Buscar no Pinecone pelos embeddings mais próximos
        matches = self.embedding_store.search(query_embedding, top_k=top_k)

        # Verificar se houve matches
        if not matches:
            print("Nenhum match encontrado para a consulta.")
            return None, None

        # Recuperar os chunks relevantes com base nos IDs retornados
        try:
            relevant_chunks = [self.chunks[int(match['id'])] for match in matches if 'id' in match]
        except KeyError as e:
            print(f"Erro ao acessar os IDs dos chunks: {e}")
            return None, None

        if not relevant_chunks:
            print("Nenhum chunk relevante encontrado com base nos IDs retornados.")
            return None, None

        # Concatenar os chunks para formar o contexto
        context = "\n".join(relevant_chunks)

        # Criar o prompt para o LLM
        prompt = f"Aqui estão algumas informações relevantes extraídas de documentos:\n{context}\n\nCom base nessas informações, responda à seguinte pergunta:\n{user_query}"

        # Gerar a resposta com o LLM
        answer = self.llm.generate_response(prompt)

        # Avaliar a resposta se houver uma resposta de referência
        if reference_answer:
            evaluation_results = self.evaluator.evaluate(answer, reference_answer)
            print(f"Resultados da Avaliação: {evaluation_results}")

        return answer, relevant_chunks
