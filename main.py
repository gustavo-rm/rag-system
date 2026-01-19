import os
import logging
from dotenv import load_dotenv
from src.utils.logger import setup_logging

# --- Configuração de Logging ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Carregamento de Variáveis ---
load_dotenv()

# --- CORREÇÃO DE TIMEOUT ---
# Força o uso de arquivos em cache local, evitando conexões com HuggingFace
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- Importações ---

# Ingestion
from src.ingestion.chunker import Chunker

# Core AI Components
from src.components.embedder import Embedder
from src.components.reranker import ReRanker
from src.components.llm import LLM, LLMGenerationError
from src.components.hybrid_retriever import HybridRetriever
from src.routing.query_router import QueryRouter

# Query Transformers
from src.query_transformers import NoOpTransformer, MultiQueryTransformer, HyDETransformer

# Caching & Preprocessing
from src.caching.cache_manager import CacheManager
from src.caching.semantic_cache import SemanticCache
from src.preprocessing.query_corrector import QueryCorrector

# Storage
from src.stores import get_vector_store

# Application Logic
from src.pipeline import RAGSystem
from src.chat.chatbot import Chatbot


def main():
    """
    Função principal de orquestração do RAG.
    """
    logger.info("🚀 Inicializando sistema RAG...")

    # ==========================================
    # 1. CONFIGURAÇÃO DOS COMPONENTES BASE
    # ==========================================

    # --- A. Vector Store (Banco de Dados) ---
    logger.info("Configurando Vector Store...")
    config_store = {
        'type': 'chroma',
        'path': 'data/chromaDB/',
        'collection_name': 'rag_project_v2'
    }
    base_vector_store = get_vector_store(config_store)

    # --- B. Retriever (Híbrido) ---
    # Envolve o banco vetorial para adicionar capacidade de busca por palavra-chave (BM25)
    hybrid_retriever = HybridRetriever(base_vector_store)

    # --- C. Componentes de IA (Embedder, LLM, ReRanker) ---

    # Chunker
    chunker = Chunker(chunk_size=512, chunk_overlap=50)

    # --- Configuração do Modelo de Embedding ---

    # Caminho onde o script de treino salvou o modelo
    finetuned_model_path = "models/finetuned_v3"
    base_model_name = "paraphrase-multilingual-mpnet-base-v2"

    # Lógica inteligente de seleção
    if os.path.exists(finetuned_model_path):
        logger.info(f"💎 Modelo Fine-Tuned detectado! Usando: {finetuned_model_path}")
        selected_model = finetuned_model_path
    else:
        logger.warning(
            f"⚠️ Modelo Fine-Tuned não encontrado em '{finetuned_model_path}'. Usando modelo base: {base_model_name}")
        selected_model = base_model_name

    # Instancia o Embedder com o modelo escolhido
    embedder = Embedder(
        method='sbert',
        model_name=selected_model
        # batch_size será auto-configurado (32 para GPU)
    )

    # ReRanker: Atualizado para modelo BAAI (Melhor suporte a Multilíngue/PT-BR)
    reranker = ReRanker(model_name='BAAI/bge-reranker-base')

    # LLM: Configurado com controle de Context Window e No-Grad
    # Se usar OpenAI, lembrar de configurar a key no .env
    llm = LLM(
        method='local',
        model_name='microsoft/Phi-3-mini-4k-instruct',
        context_window=4096  # Limite do Phi-3
    )

    # ==========================================
    # 2. ESTRATÉGIAS DE ROTEAMENTO (ROUTER)
    # ==========================================
    logger.info("Configurando estratégias de Query Routing...")

    # Instancia as estratégias injetando o LLM onde necessário
    transformers_map = {
        "noop": NoOpTransformer(),
        "hyde": HyDETransformer(llm),
        "multi_query": MultiQueryTransformer(llm, num_queries=3)
    }

    # O Router recebe o mapa e decidirá qual usar em tempo de execução
    query_router = QueryRouter(llm, strategies=transformers_map)

    # ==========================================
    # 3. CACHE E PRÉ-PROCESSAMENTO
    # ==========================================

    # Pega dimensão dinamicamente do modelo carregado (ex: 768 para mpnet)
    # Acessamos o atributo interno do SentenceTransformer se for método sbert
    embedding_dim = 768  # Valor padrão seguro para mpnet-base
    if hasattr(embedder, 'model') and hasattr(embedder.model, 'get_sentence_embedding_dimension'):
        embedding_dim = embedder.model.get_sentence_embedding_dimension()

    exact_cache = CacheManager()
    semantic_cache = SemanticCache(dimension=embedding_dim, similarity_threshold=0.92)
    query_corrector = QueryCorrector(language='pt', enable_grammar=True)

    # ==========================================
    # 4. MONTAGEM DO SISTEMA (RAG + CHATBOT)
    # ==========================================

    logger.info("Montando Pipeline RAG...")

    rag_system = RAGSystem(
        chunker=chunker,
        embedder=embedder,
        retriever=hybrid_retriever,
        reranker=reranker,
        llm=llm,
        router=query_router
    )

    chatbot = Chatbot(
        llm=llm,
        rag_system=rag_system,
        cache_manager=exact_cache,
        semantic_cache=semantic_cache,
        query_corrector=query_corrector
    )

    # ==========================================
    # 5. INGESTÃO DE DADOS (Execução Única)
    # ==========================================

    pdf_path = "data/pdfs/relevo-brasileiro.pdf"

    # Verifica se o arquivo existe
    if os.path.exists(pdf_path):
        # Lógica simples para evitar re-ingestão a cada boot
        # Em produção, você verificaria se o arquivo já está no banco pelo hash ou nome
        ingestion_done_marker = f"{pdf_path}.done"

        if not os.path.exists(ingestion_done_marker):
            logger.info(f"Iniciando ingestão do documento: {pdf_path}")
            try:
                rag_system.setup_pipeline(pdf_path)
                # Cria um arquivo vazio para marcar que já foi feito
                with open(ingestion_done_marker, 'w') as f:
                    f.write('done')
                logger.info("Ingestão concluída e marcada.")
            except Exception as e:
                logger.error(f"Falha na ingestão: {e}")
        else:
            logger.info("Documento já processado anteriormente. Pulando ingestão.")
    else:
        logger.warning(f"PDF não encontrado em '{pdf_path}'. O sistema funcionará apenas com conhecimento prévio.")

    # ==========================================
    # 6. LOOP DE INTERAÇÃO (CHAT)
    # ==========================================

    print("\n" + "=" * 50)
    print("🤖 Assistente RAG v2.0 Pronto!")
    print("Comandos: 'sair' para encerrar.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_question = input("Você: ").strip()

            if not user_question:
                continue

            if user_question.lower() in ['sair', 'exit', 'quit']:
                logger.info("Encerrando sessão.")
                print("Assistente: Até logo! 👋")
                break

            # O Chatbot gerencia todo o fluxo (correção -> cache -> RAG -> Resposta)
            response = chatbot.chat(user_question)

            print(f"Assistente: {response}\n")

        except KeyboardInterrupt:
            print("\nOperação cancelada pelo usuário.")
            break
        except LLMGenerationError as e:
            logger.error(f"Erro no LLM: {e}")
            print("Assistente: Desculpe, tive um problema ao gerar a resposta. Tente simplificar a pergunta.")
        except Exception as e:
            logger.critical(f"Erro não tratado: {e}")
            print("Assistente: Ocorreu um erro interno.")


if __name__ == "__main__":
    main()