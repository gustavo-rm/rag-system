import logging
from typing import Dict, Any, List

from src.ingestion.chunker import Chunker
from src.components.embedder import Embedder
from src.components.llm import LLM, LLMGenerationError
from src.ingestion.pdf_processor import PDFProcessor
from src.components.reranker import ReRanker
from src.components.hybrid_retriever import HybridRetriever
from src.routing.query_router import QueryRouter

# Configuração de Logger para o módulo
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Orquestrador principal do pipeline RAG (Retrieval-Augmented Generation).

    Esta classe atua como o controlador central, integrando os componentes de ingestão,
    indexação, recuperação, reclassificação e geração. Ela abstrai a complexidade
    do fluxo de dados para o chatbot ou API.

    Fluxo de Processamento (Método `ask`):
    1. **Roteamento:** O `QueryRouter` analisa a pergunta e escolhe a melhor estratégia (ex: HyDE).
    2. **Expansão:** A estratégia escolhida transforma a pergunta em uma ou mais queries otimizadas.
    3. **Recuperação Híbrida:** O `HybridRetriever` busca candidatos usando BM25 (palavra-chave) e Embeddings (semântica).
    4. **Reclassificação:** O `ReRanker` (Cross-Encoder) filtra e ordena os melhores documentos com precisão.
    5. **Geração:** O `LLM` recebe o contexto refinado e gera a resposta final.

    Attributes:
        chunker (Chunker): Responsável pela divisão de textos longos.
        embedder (Embedder): Responsável pela vetorização de textos.
        retriever (HybridRetriever): Gerencia o banco vetorial e busca por palavras-chave.
        reranker (ReRanker): Refina a relevância dos documentos recuperados.
        llm (LLM): Modelo de linguagem para geração de respostas e roteamento.
        router (QueryRouter): Cérebro que decide a estratégia de busca dinamicamente.
    """

    def __init__(self, chunker: Chunker, embedder: Embedder, retriever: HybridRetriever,
                 reranker: ReRanker, llm: LLM, router: QueryRouter):
        """
        Inicializa o sistema RAG injetando todas as dependências necessárias.

        Args:
            chunker (Chunker): Componente de divisão de texto.
            embedder (Embedder): Componente de geração de embeddings.
            retriever (HybridRetriever): Componente de busca híbrida (BM25 + Vetor).
            reranker (ReRanker): Componente de reclassificação (Cross-Encoder).
            llm (LLM): Interface do modelo de linguagem.
            router (QueryRouter): Componente de decisão de estratégia de busca.
        """
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.router = router

        # Prompt de sistema (System Message) fixo para garantir comportamento do Assistente
        self.system_prompt = """Você é um assistente especialista e atencioso. Sua tarefa é responder à pergunta do usuário estritamente com base no contexto fornecido.
                            Regras:
                            1. Analise o contexto e a pergunta cuidadosamente.
                            2. Responda de forma concisa e direta, usando apenas as informações encontradas no contexto.
                            3. Se a resposta não estiver no contexto, responda exatamente: 'A informação não foi encontrada nos documentos fornecidos.'
                            4. Não adicione nenhuma informação externa ou conhecimento prévio."""

    def setup_pipeline(self, pdf_path: str):
        """
        Executa o pipeline de ingestão de documentos (ETL: Extract, Transform, Load).

        O processo envolve:
        1. Extração e limpeza do texto do PDF.
        2. Divisão em chunks (pedaços) com sobreposição.
        3. Geração de embeddings para cada chunk.
        4. Indexação no VectorStore e no índice BM25 (HybridRetriever).

        Args:
            pdf_path (str): Caminho absoluto ou relativo para o arquivo PDF.

        Raises:
            FileNotFoundError: Se o arquivo PDF não existir.
            Exception: Para erros gerais de processamento ou conexão com banco de dados.
        """
        logger.info(f"--- Iniciando pipeline de ingestão para: {pdf_path} ---")

        try:
            processor = PDFProcessor(pdf_path)
            text = processor.extract_text()
            logger.info(f"Texto extraído e limpo. Total de caracteres: {len(text)}")

            chunks = self.chunker.chunk_text(text)
            logger.info(f"Texto dividido em {len(chunks)} chunks.")

            embeddings = self.embedder.generate_embeddings(chunks)
            logger.info(f"Embeddings gerados para todos os chunks.")

            # Adiciona ao retriever (Salva no Chroma/Pinecone E atualiza índice BM25 em memória)
            self.retriever.add_documents(chunks, embeddings)
            logger.info("--- Pipeline de ingestão concluído com sucesso! ---")

        except Exception as e:
            logger.error(f"Falha crítica durante a ingestão do arquivo '{pdf_path}': {e}")
            raise e

    def ask(self, question: str, retrieval_top_k: int = 20, rerank_top_n: int = 3) -> Dict[str, Any]:
        """
        Processa uma pergunta do usuário ponta a ponta.

        Este método orquestra a inteligência do sistema, desde a compreensão da pergunta
        até a geração da resposta fundamentada.

        Args:
            question (str): A pergunta original do usuário.
            retrieval_top_k (int, opcional): Número de documentos candidatos a recuperar na etapa 1 (Busca Híbrida).
                                             Recomenda-se um valor alto (ex: 20-50) para garantir revocação. Padrão: 20.
            rerank_top_n (int, opcional): Número de documentos finais a enviar para o LLM após a etapa 2 (Re-ranking).
                                          Deve ser baixo (ex: 3-5) para caber no contexto do LLM. Padrão: 3.

        Returns:
            Dict[str, Any]: Um dicionário contendo os resultados da execução:
                - 'question': A pergunta original.
                - 'answer': A resposta gerada pelo LLM.
                - 'contexts': Lista de strings com os textos usados como base.
                - 'strategy_used': Nome da estratégia escolhida pelo Router (ex: 'HyDETransformer').

        Raises:
            LLMGenerationError: Se houver falha na comunicação com o LLM.
            Exception: Para erros inesperados no pipeline.
        """
        logger.info(f"--- Nova Pergunta: {question} ---")

        # 1. ROTEAMENTO DINÂMICO
        # O Router decide se usa HyDE, MultiQuery ou busca direta (NoOp)
        selected_strategy = self.router.route(question)
        logger.debug(f"Estratégia selecionada: {selected_strategy.__class__.__name__}")

        # 2. TRANSFORMAÇÃO DA CONSULTA
        transformed_queries = selected_strategy.transform(question)

        # 3. EMBEDDING DAS QUERIES
        # Gera vetores para todas as variações da pergunta (ou doc hipotético)
        query_embeddings = self.embedder.generate_embeddings(transformed_queries)

        # 4. RECUPERAÇÃO HÍBRIDA (Estágio 1 - Alta Revocação)
        logger.info(f"Recuperando candidatos para {len(transformed_queries)} variações de query...")
        all_candidate_docs = []

        for query_text, query_vec in zip(transformed_queries, query_embeddings):
            # Busca Híbrida: Combina scores de vetores (semântico) e BM25 (palavra-chave)
            results = self.retriever.search(
                query_text=query_text,
                query_embedding=query_vec,
                top_k=retrieval_top_k
            )
            all_candidate_docs.extend(results)

        # Desduplicação de documentos (pois MultiQuery pode retornar o mesmo doc várias vezes)
        unique_docs_dict = {}
        for doc in all_candidate_docs:
            doc_id = doc['id']
            # Prioriza documentos novos ou com scores (vetoriais) mais altos
            if doc_id not in unique_docs_dict:
                unique_docs_dict[doc_id] = doc
            elif doc.get('score', 0) > unique_docs_dict[doc_id].get('score', 0):
                unique_docs_dict[doc_id] = doc

        candidate_docs = list(unique_docs_dict.values())
        logger.debug(f"Candidatos únicos recuperados: {len(candidate_docs)}")

        # 5. RE-RANKING (Estágio 2 - Alta Precisão)
        # O Cross-Encoder analisa profundamente a relação Pergunta <-> Documento
        logger.info(f"Reclassificando {len(candidate_docs)} documentos...")
        reranked_docs = self.reranker.rerank(question, candidate_docs, top_n=rerank_top_n)

        retrieved_contexts = [doc['metadata']['text'] for doc in reranked_docs]
        logger.info(f"Contextos finais selecionados para geração: {len(retrieved_contexts)}")

        # 6. GERAÇÃO (LLM)
        context_str = "\n\n---\n\n".join(retrieved_contexts)
        user_prompt = f"""
        [CONTEXTO]
        {context_str}
        [/CONTEXTO]

        Com base estritamente no contexto acima, responda à seguinte pergunta:
        Pergunta: {question}
        """

        answer = self.llm.generate_response(
            prompt=user_prompt,
            system_prompt=self.system_prompt
        )
        logger.info("Resposta gerada com sucesso.")

        return {
            "question": question,
            "answer": answer,
            "contexts": retrieved_contexts,
            "strategy_used": selected_strategy.__class__.__name__
        }
