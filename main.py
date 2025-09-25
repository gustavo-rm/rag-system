import os
from dotenv import load_dotenv

load_dotenv()

from src.chunker import Chunker
from src.embedder import Embedder
from src.stores import get_vector_store
from src.llm import LLM
from src.reranker import ReRanker
from src.rag_system import RAGSystem


def main():
    # --- 1. Configuração dos Componentes ---

    config_chroma = {
        'type': 'chroma',
        'path': 'data/chromaDB/',
        'collection_name': 'rag_project'
    }
    vector_store = get_vector_store(config_chroma)

    # Inicialização dos outros componentes
    chunker = Chunker(chunk_size=512, chunk_overlap=50)
    embedder = Embedder(method='sbert', model_name='paraphrase-multilingual-mpnet-base-v2')
    reranker = ReRanker()  # <-- Instancia o novo componente
    llm = LLM(method='local')

    # --- 2. Montagem do Sistema RAG ---
    rag_system = RAGSystem(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        llm=llm
    )

    # --- 3. Execução do Pipeline ---
    pdf_path = "data/pdfs/relevo-brasileiro.pdf"
    if os.path.exists(pdf_path):
        # A linha abaixo pode ser comentada após a primeira execução para não reprocessar o mesmo PDF
        rag_system.setup_pipeline(pdf_path)
        pass
    else:
        print(f"Arquivo PDF não encontrado em '{pdf_path}'. Crie um para continuar.")
        return

    # --- 4. Realizando Perguntas ---
    while True:
        question = input("\nFaça sua pergunta (ou digite 'sair' para terminar): ")
        if question.lower() == 'sair':
            break

        response = rag_system.ask(question)

        print("\n--- Contextos Utilizados (após re-ranking) ---")
        for i, context in enumerate(response['contexts']):
            print(f"[{i + 1}] {context[:150]}...")
        print("--------------------------")


if __name__ == "__main__":
    main()