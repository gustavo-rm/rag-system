from src.rag_system import RAGSystem
from dotenv import load_dotenv
import os

# Carregar as variáveis de ambiente do .env
load_dotenv()

# Carregar as chaves de API das variáveis de ambiente
var_pinecone_api_key = os.getenv("PINECONE_API_KEY")
var_pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
var_openai_api_key = os.getenv("OPENAI_API_KEY")

def main():
    # Definir o caminho do PDF
    pdf_path = "data/pdfs/relevo-brasileiro.pdf"

    # Inicializar o sistema RAG com as configurações necessárias
    rag_system = RAGSystem(
        pdf_path=pdf_path,
        chunk_method="sentences",  # Chunking baseado em sentenças
        chunk_size=100,  # Tamanho dos chunks
        embedder_method="sbert",  # Usar OpenAI ou Sentence-BERT
        openai_api_key=var_openai_api_key,
        pinecone_api_key=var_pinecone_api_key,
        pinecone_environment=var_pinecone_environment,
        embedding_dimension=384,
        index_name="my-vector-index",  # Nome do índice Pinecone
        llm_method="local"  # Usar OpenAI para LLM
    )

    # Etapa 1: Preparar os dados (extrair texto do PDF, chunkar e armazenar embeddings no Pinecone)
    print("Preparando os dados...")
    rag_system.prepare_data()
    print("Dados preparados com sucesso!")

    # Etapa 2: Receber a consulta do usuário
    user_query = input("Digite sua pergunta: ")

    # Etapa 3: Usar o sistema RAG para buscar e gerar uma resposta
    print("Buscando resposta...")
    answer, relevant_chunks = rag_system.query(user_query)

    # Exibir a resposta final
    print("\nResposta Final:")
    print(answer)

    # Exibir os chunks relevantes que foram usados para gerar a resposta
    print("\nChunks Relevantes Utilizados:")
    for i, chunk in enumerate(relevant_chunks):
        print(f"Chunk {i+1}: {chunk}")

if __name__ == "__main__":
    main()
