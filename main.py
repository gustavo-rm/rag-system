import os
from dotenv import load_dotenv

from src.components.hybrid_retriever import HybridRetriever
from src.preprocessing.query_corrector import QueryCorrector

# --- Carregamento das Variáveis de Ambiente ---
# Carrega chaves de API e outras configurações do arquivo .env
load_dotenv()

# --- Importação dos Componentes da Arquitetura ---
# Ingestion Pipeline
from src.ingestion.chunker import Chunker

# Core AI Components
from src.components.embedder import Embedder
from src.components.reranker import ReRanker
from src.components.llm import LLM
from src.caching.cache_manager import CacheManager
from src.caching.semantic_cache import SemanticCache

# Storage Backend
from src.stores import get_vector_store

# Query Transformation Strategies
from src.query_transformers import NoOpTransformer, MultiQueryTransformer, HyDETransformer

# Main Application Logic
from src.pipeline import RAGSystem
from src.chat.chatbot import Chatbot

def main():
    """
    Função principal que configura e executa o Chatbot RAG.
    """
    print("--- 1. CONFIGURAÇÃO DOS COMPONENTES ---")

    # --- Configuração do Armazenamento Vetorial (Vector Store) ---
    # Escolha entre 'chroma' (local) ou 'pinecone' (nuvem)
    config_store = {
        'type': 'chroma',
        'path': 'data/chromaDB/',
        'collection_name': 'rag_project'
    }
    base_vector_store = get_vector_store(config_store)

    # Cria o HybridRetriever envolvendo o store
    hybrid_retriever = HybridRetriever(base_vector_store)

    # --- Configuração dos Componentes de Ingestão e IA ---
    chunker = Chunker(chunk_size=512, chunk_overlap=50)

    # Use um modelo genérico ou o seu modelo treinado
    # finetuned_model_path = './models/finetuned-embedder'
    embedder = Embedder(method='sbert', model_name='paraphrase-multilingual-mpnet-base-v2')
    # embedder = Embedder(method='sbert', model_name=finetuned_model_path) # Para usar o modelo treinado

    reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
    llm = LLM(method='local', model_name='microsoft/Phi-3-mini-4k-instruct')

    # Para usar OpenAI, descomente a linha abaixo e configure a API_KEY no .env
    # llm = LLM(method='openai', model_name='gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))

    # --- Configuração das Estratégias de Otimização ---
    # --- Criar um dicionário com todas as estratégias de transformação disponíveis ---
    available_transformers = {
        "NoOpTransformer": NoOpTransformer(),
        "HyDETransformer": HyDETransformer(llm=llm),
        "MultiQueryTransformer": MultiQueryTransformer(llm=llm, num_queries=3)
    }

    # Configuração do Cache de Duas Camadas
    embedding_dimension = embedder.model.get_sentence_embedding_dimension()
    exact_cache = CacheManager()
    semantic_cache = SemanticCache(dimension=embedding_dimension, similarity_threshold=0.92)

    # Instancia o componente de pré-processamento
    query_corrector = QueryCorrector(language='pt')

    print("\n--- 2. MONTAGEM DOS SISTEMAS ---")

    # O RAGSystem é a base de conhecimento que responde a perguntas autônomas
    rag_system = RAGSystem(
        chunker=chunker,
        embedder=embedder,
        retriever=hybrid_retriever,
        reranker=reranker,
        llm=llm,
        query_transformer=available_transformers["NoOpTransformer"]
    )

    # O Chatbot é a camada de conversação que gerencia o histórico e o cache
    chatbot = Chatbot(
        llm=llm,
        rag_system=rag_system,
        cache_manager=exact_cache,
        semantic_cache=semantic_cache,
        transformers=available_transformers,
        query_corrector=query_corrector
    )


    print("\n--- 3. PIPELINE DE INGESTÃO DE DADOS ---")

    pdf_path = "data/pdfs/relevo-brasileiro.pdf"
    if os.path.exists(pdf_path):
        # A linha abaixo deve ser executada apenas uma vez por documento
        # para processar e armazenar seus embeddings.
        # Após a primeira execução, comente-a para não reprocessar desnecessariamente.
        # rag_system.setup_pipeline(pdf_path)
        print(f"Sistema pronto para consultar o documento: {pdf_path}")
        pass # Comente esta linha e descomente a de cima para a ingestão
    else:
        print(f"AVISO: Arquivo PDF não encontrado em '{pdf_path}'. O sistema só responderá com o conhecimento geral do LLM.")


    print("\n--- 4. INICIANDO O CHAT INTERATIVO ---")
    print("\n\nAssistente de Documentos iniciado! Faça sua pergunta.")
    print("Digite 'sair' para terminar a conversa.")

    while True:
        user_question = input("\nVocê: ")
        if user_question.lower() in ['sair', 'exit', 'quit']:
            print("Assistente: Até logo!")
            break

        assistant_response = chatbot.chat(user_question)
        print(f"Assistente: {assistant_response}")


if __name__ == "__main__":
    main()