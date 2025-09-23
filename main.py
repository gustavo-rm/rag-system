import os

from dotenv import load_dotenv

# Carrega as variáveis de ambiente (chaves de API, etc.)
load_dotenv()

# Importa todos os nossos componentes e a fábrica do Vector Store
from src.chunker import Chunker
from src.embedder import Embedder
from src.stores import get_vector_store
from src.llm import LLM
from src.rag_system import RAGSystem


def main():
    # --- 1. Configuração dos Componentes ---

    # Configuração do Vector Store (usando ChromaDB local)
    config_chroma = {
        'type': 'chroma',
        'path': 'data/chromaDB/',
        'collection_name': 'rag_project'
    }
    vector_store = get_vector_store(config_chroma)

    # Inicialização dos outros componentes
    chunker = Chunker(chunk_size=512, chunk_overlap=50)
    embedder = Embedder(method='sbert', model_name='paraphrase-multilingual-mpnet-base-v2')

    # LLM local (padrão: Phi-3-mini)
    llm = LLM(method='local')

    # Para usar OpenAI, descomente a linha abaixo e configure a API_KEY no .env
    # llm = LLM(method='openai', model_name='gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))

    # --- 2. Montagem do Sistema RAG ---
    rag_system = RAGSystem(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm
    )

    # --- 3. Execução do Pipeline ---

    # Limpar dados antigos (opcional, bom para testes)
    # vector_store.delete() 

    # Ingestão de um novo documento PDF
    pdf_path = "data/pdfs/relevo-brasileiro.pdf"  # <-- SUBSTITUA PELO CAMINHO DO SEU PDF
    if os.path.exists(pdf_path):
        rag_system.setup_pipeline(pdf_path)
    else:
        print(f"Arquivo PDF não encontrado em '{pdf_path}'. Crie um para continuar.")
        return

    # --- 4. Realizando Perguntas ---

    # Loop interativo para fazer perguntas
    while True:
        question = input("\nFaça sua pergunta (ou digite 'sair' para terminar): ")
        if question.lower() == 'sair':
            break

        response = rag_system.ask(question)

        # Imprimir a resposta e os contextos usados
        print("\n--- Contextos Utilizados ---")
        for i, context in enumerate(response['contexts']):
            print(f"[{i + 1}] {context[:150]}...")
        print("--------------------------")


if __name__ == "__main__":
    main()