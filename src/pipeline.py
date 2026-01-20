import logging
from typing import Dict, Any, List, Optional

from src.ingestion.chunker import Chunker
from src.components.embedder import Embedder
from src.components.llm import LLM
from src.ingestion.pdf_processor import PDFProcessor
from src.components.reranker import ReRanker
from src.components.hybrid_retriever import HybridRetriever
from src.routing.query_router import QueryRouter

# Configuração de Logger para o módulo
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Orquestrador principal do pipeline RAG (Retrieval-Augmented Generation).

    Esta classe atua como o controlador central (Facade), integrando os componentes de ingestão,
    indexação, recuperação, reclassificação e geração.

    Fluxo de Processamento (Método `ask`):
    1. **Roteamento:** O `QueryRouter` decide a estratégia.
    2. **Recuperação:** O método `_retrieve_candidates` busca e deduplica documentos.
    3. **Refinamento:** O método `_rerank_contexts` filtra os melhores textos.
    4. **Geração:** O método `_generate_answer` produz a resposta final.

    Attributes:
        chunker (Chunker): Divisão de textos.
        embedder (Embedder): Vetorização.
        retriever (HybridRetriever): Busca (Sparse + Dense).
        reranker (ReRanker): Refinamento de precisão.
        llm (LLM): Geração e Raciocínio.
        router (QueryRouter): Decisão de estratégia.
    """

    def __init__(self, chunker: Chunker, embedder: Embedder, retriever: HybridRetriever,
                 reranker: ReRanker, llm: LLM, router: QueryRouter):
        """
        Inicializa o sistema RAG injetando dependências.
        """
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.router = router

        self.system_prompt = """
        Você é um assistente geográfico e técnico preciso.
        Sua única fonte de verdade são os [CONTEXTOS] fornecidos abaixo.

        Diretrizes:
        1. Responda à pergunta do usuário usando APENAS as informações do contexto.
        2. Responda em Português do Brasil de forma fluida e direta.
        3. Se o contexto contiver a resposta, explique-a detalhadamente.
        4. Se o contexto mencionar o assunto mas não tiver a resposta exata, diga o que encontrou sobre o tema.
        5. SOMENTE se o contexto for totalmente irrelevante, diga: "A informação não foi encontrada nos documentos fornecidos."
        """

    def setup_pipeline(self, pdf_path: str):
        """
        Executa o pipeline de ingestão (ETL).

        Args:
            pdf_path (str): Caminho para o PDF.
        """
        logger.info(f"--- Iniciando pipeline de ingestão para: {pdf_path} ---")

        try:
            processor = PDFProcessor(pdf_path)
            text = processor.extract_text()
            logger.info(f"Texto extraído. Caracteres: {len(text)}")

            # O Chunker novo já limpa espaços extras
            chunks = self.chunker.chunk_text(text)
            logger.info(f"Texto dividido em {len(chunks)} chunks.")

            embeddings = self.embedder.generate_embeddings(chunks)
            logger.info(f"Embeddings gerados.")

            # Salva no banco e na memória do BM25
            self.retriever.add_documents(chunks, embeddings)
            logger.info("--- Pipeline de ingestão concluído! ---")

        except Exception as e:
            logger.error(f"❌ Falha crítica na ingestão: {e}")
            raise e

    def ask(self, question: str, retrieval_top_k: int = 20, rerank_top_n: int = 5) -> Dict[str, Any]:
        """
        Orquestra o fluxo de resposta para uma pergunta do usuário.

        Args:
            question (str): Pergunta do usuário.
            retrieval_top_k (int): Documentos para recuperar na fase 1.
            rerank_top_n (int): Documentos finais para o LLM.

        Returns:
            Dict com a resposta e metadados.
        """
        logger.info(f"--- Nova Pergunta: {question} ---")

        # 1. Roteamento e Transformação
        selected_strategy = self.router.route(question)
        logger.debug(f"Estratégia: {selected_strategy.__class__.__name__}")
        transformed_queries = selected_strategy.transform(question)

        # 2. Recuperação e Deduplicação (Retrieval)
        candidate_docs = self._retrieve_candidates(transformed_queries, retrieval_top_k)

        if not candidate_docs:
            logger.warning("Nenhum documento encontrado na fase de Retrieval.")
            return self._build_empty_response(question, selected_strategy)

        # 3. Reclassificação e Seleção (Reranking)
        final_contexts = self._rerank_contexts(question, candidate_docs, rerank_top_n)

        if not final_contexts:
            logger.warning("O ReRanker filtrou todos os documentos!")
            return self._build_empty_response(question, selected_strategy)

        # 4. Geração (LLM)
        answer = self._generate_answer(question, final_contexts)

        return {
            "question": question,
            "answer": answer,
            "contexts": final_contexts,
            "strategy_used": selected_strategy.__class__.__name__
        }

    # --- MÉTODOS AUXILIARES (MODULARIZAÇÃO) ---

    def _retrieve_candidates(self, queries: List[str], top_k: int) -> List[Dict[str, Any]]:
        """
        Executa a busca híbrida para múltiplas variações de query e deduplica os resultados.

        Args:
            queries (List[str]): Lista de variações da pergunta.
            top_k (int): Limite de documentos por query.

        Returns:
            List[Dict]: Lista única de documentos candidatos (dicts com metadados).
        """
        query_embeddings = self.embedder.generate_embeddings(queries)
        logger.info(f"Recuperando candidatos para {len(queries)} variações de query...")

        all_candidate_docs = []

        # Busca Híbrida para cada variação
        for query_text, query_vec in zip(queries, query_embeddings):
            results = self.retriever.search(
                query_text=query_text,
                query_embedding=query_vec,
                top_k=top_k
            )
            all_candidate_docs.extend(results)

        # Deduplicação Inteligente (Mantém o documento com maior score encontrado)
        unique_docs_map = {}
        for doc in all_candidate_docs:
            doc_id = str(doc.get('id', 'unknown'))
            current_score = doc.get('score', 0)

            if doc_id not in unique_docs_map:
                unique_docs_map[doc_id] = doc
            else:
                if current_score > unique_docs_map[doc_id].get('score', 0):
                    unique_docs_map[doc_id] = doc

        unique_docs = list(unique_docs_map.values())
        logger.debug(f"Candidatos únicos recuperados: {len(unique_docs)}")
        return unique_docs

    def _rerank_contexts(self, question: str, candidate_docs: List[Dict[str, Any]], top_n: int) -> List[str]:
        """
        Extrai o texto dos documentos e aplica o Cross-Encoder para ordenar por relevância.

        Args:
            question (str): A pergunta original.
            candidate_docs (List[Dict]): Documentos vindos do retriever.
            top_n (int): Quantidade final de contextos.

        Returns:
            List[str]: Lista de strings (apenas o texto) dos melhores contextos.
        """
        # Extração segura do texto dos metadados
        candidate_texts = [
            doc['metadata']['text']
            for doc in candidate_docs
            if 'metadata' in doc and 'text' in doc['metadata']
        ]

        if not candidate_texts:
            return []

        logger.info(f"Reclassificando {len(candidate_texts)} documentos...")

        # O reranker retorna strings ordenadas
        final_contexts = self.reranker.rerank(question, candidate_texts, top_n=top_n)
        logger.info(f"Contextos finais selecionados: {len(final_contexts)}")

        # --- LOG VISUAL (RAIO-X) ---
        if final_contexts:
            logger.info("📝 --- INÍCIO DO CONTEXTO ENVIADO AO LLM ---")
            for i, ctx in enumerate(final_contexts):
                preview = ctx.replace('\n', ' ')[:200]
                logger.info(f"📜 [Chunk {i + 1}]: \"{preview}...\"")
            logger.info("📝 --- FIM DO CONTEXTO ---")

        return final_contexts

    def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """
        Monta o prompt final e chama o LLM.

        Args:
            question (str): Pergunta do usuário.
            contexts (List[str]): Contextos validados.

        Returns:
            str: Resposta gerada.
        """
        context_block = "\n\n---\n\n".join(contexts)

        user_prompt = f"""
        [CONTEXTOS RECUPERADOS]
        {context_block}

        [PERGUNTA DO USUÁRIO]
        {question}

        Com base estritamente nos contextos acima, qual a resposta?
        """

        try:
            return self.llm.generate_response(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"Erro na geração do LLM: {e}")
            return "Ocorreu um erro ao gerar a resposta."

    def _build_empty_response(self, question: str, strategy: Any) -> Dict[str, Any]:
        """Helper para retornar resposta padrão quando nada é encontrado."""
        return {
            "question": question,
            "answer": "A informação não foi encontrada nos documentos fornecidos.",
            "contexts": [],
            "strategy_used": strategy.__class__.__name__
        }