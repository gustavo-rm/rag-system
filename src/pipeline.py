import logging
from typing import Dict, Any, List, Optional

from src.ingestion.chunker import Chunker
from src.components.embedder import Embedder
from src.components.llm import LLM
from src.ingestion.pdf_processor import PDFProcessor
from src.components.reranker import ReRanker
from src.components.hybrid_retriever import HybridRetriever
from src.routing.query_router import QueryRouter
from src.utils.exceptions import IngestionError, EmbeddingError, VectorStoreError

# Logger Configuration for the module
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Main Orchestrator of the RAG (Retrieval-Augmented Generation) pipeline.

    This class acts as the central controller (Facade), integrating the components of ingestion,
    indexing, retrieval, reranking, and generation.

    Processing Flow (Method `ask`):
    1. **Routing:** The `QueryRouter` decides the strategy.
    2. **Retrieval:** The `_retrieve_candidates` method searches and deduplicates documents.
    3. **Refinement:** The `_rerank_contexts` method filters the best texts.
    4. **Generation:** The `_generate_answer` method produces the final answer.

    Attributes:
        chunker (Chunker): Text division.
        embedder (Embedder): Vectorization.
        retriever (HybridRetriever): Search (Sparse + Dense).
        reranker (ReRanker): Precision refinement.
        llm (LLM): Generation and Reasoning.
        router (QueryRouter): Strategy decision.
    """

    def __init__(self, chunker: Chunker, embedder: Embedder, retriever: HybridRetriever,
                 reranker: ReRanker, llm: LLM, router: QueryRouter):
        """
        Initializes the RAG system by injecting dependencies.

        Args:
            chunker (Chunker): Component for splitting text into chunks.
            embedder (Embedder): Component for generating embeddings.
            retriever (HybridRetriever): Component for hybrid search.
            reranker (ReRanker): Component for re-ranking results.
            llm (LLM): Component for text generation.
            router (QueryRouter): Component for routing queries.
        """
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.router = router

        self.system_prompt = """
        You are a precise geographical and technical assistant.
        Your only source of truth are the [CONTEXTS] provided below.

        Guidelines:
        1. Answer the user's question using ONLY the information from the context.
        2. Answer in Brazilian Portuguese in a fluid and direct manner.
        3. If the context contains the answer, explain it in detail.
        4. If the context mentions the subject but doesn't have the exact answer, say what you found about the topic.
        5. ONLY if the context is totally irrelevant, say: "The information was not found in the provided documents."
        """

    def setup_pipeline(self, pdf_path: str):
        """
        Executes the ingestion pipeline (ETL).

        Args:
            pdf_path (str): Path to the PDF file.

        Raises:
            IngestionError: If file processing fails.
            EmbeddingError: If vectorization fails.
            VectorStoreError: If storage fails.
            Exception: For unexpected critical errors.
        """
        logger.info(f"--- Starting ingestion pipeline for: {pdf_path} ---")

        try:
            processor = PDFProcessor(pdf_path)
            text = processor.extract_text()
            logger.info(f"Text extracted. Characters: {len(text)}")

            # The new Chunker already cleans extra spaces
            chunks = self.chunker.chunk_text(text)
            logger.info(f"Text divided into {len(chunks)} chunks.")

            embeddings = self.embedder.generate_embeddings(chunks)
            logger.info(f"Embeddings generated.")

            # Saves to the database and BM25 memory
            self.retriever.add_documents(chunks, embeddings)
            logger.info("--- Ingestion pipeline completed! ---")

        except (IngestionError, EmbeddingError, VectorStoreError) as e:
            logger.error(f"❌ Pipeline component failure: {e}")
            raise e
        except Exception as e:
            logger.error(f"❌ Critical unhandled ingestion failure: {e}")
            raise e

    def ask(self, question: str, retrieval_top_k: int = 20, rerank_top_n: int = 5) -> Dict[str, Any]:
        """
        Orchestrates the response flow for a user question.

        Args:
            question (str): User question.
            retrieval_top_k (int): Documents to retrieve in phase 1.
            rerank_top_n (int): Final documents for the LLM.

        Returns:
            Dict[str, Any]: Dictionary with the answer and metadata.
        """
        logger.info(f"--- New Question: {question} ---")

        # 1. Routing and Transformation
        selected_strategy = self.router.route(question)
        logger.debug(f"Strategy: {selected_strategy.__class__.__name__}")
        transformed_queries = selected_strategy.transform(question)

        # 2. Retrieval and Deduplication
        candidate_docs = self._retrieve_candidates(transformed_queries, retrieval_top_k)

        if not candidate_docs:
            logger.warning("No documents found in Retrieval phase.")
            return self._build_empty_response(question, selected_strategy)

        # 3. Re-ranking and Selection
        final_contexts = self._rerank_contexts(question, candidate_docs, rerank_top_n)

        if not final_contexts:
            logger.warning("ReRanker filtered all documents!")
            return self._build_empty_response(question, selected_strategy)

        # 4. Generation (LLM)
        answer = self._generate_answer(question, final_contexts)

        return {
            "question": question,
            "answer": answer,
            "contexts": final_contexts,
            "strategy_used": selected_strategy.__class__.__name__
        }

    # --- HELPER METHODS (MODULARIZATION) ---

    def _retrieve_candidates(self, queries: List[str], top_k: int) -> List[Dict[str, Any]]:
        """
        Executes hybrid search for multiple query variations and deduplicates results.

        Args:
            queries (List[str]): List of query variations.
            top_k (int): Limit of documents per query.

        Returns:
            List[Dict]: Unique list of candidate documents (dicts with metadata).
        """
        query_embeddings = self.embedder.generate_embeddings(queries)
        logger.info(f"Retrieving candidates for {len(queries)} query variations...")

        all_candidate_docs = []

        # Hybrid Search for each variation
        for query_text, query_vec in zip(queries, query_embeddings):
            results = self.retriever.search(
                query_text=query_text,
                query_embedding=query_vec,
                top_k=top_k
            )
            all_candidate_docs.extend(results)

        # Smart Deduplication (Keeps the document with the highest score found)
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
        logger.debug(f"Unique candidates retrieved: {len(unique_docs)}")
        return unique_docs

    def _rerank_contexts(self, question: str, candidate_docs: List[Dict[str, Any]], top_n: int) -> List[str]:
        """
        Extracts text from documents and applies the Cross-Encoder to sort by relevance.

        Args:
            question (str): The original question.
            candidate_docs (List[Dict]): Documents coming from the retriever.
            top_n (int): Final quantity of contexts.

        Returns:
            List[str]: List of strings (text only) of the best contexts.
        """
        # Safe extraction of text from metadata
        candidate_texts = [
            doc['metadata']['text']
            for doc in candidate_docs
            if 'metadata' in doc and 'text' in doc['metadata']
        ]

        if not candidate_texts:
            return []

        logger.info(f"Re-ranking {len(candidate_texts)} documents...")

        # The reranker returns sorted strings
        final_contexts = self.reranker.rerank(question, candidate_texts, top_n=top_n)
        logger.info(f"Final contexts selected: {len(final_contexts)}")

        # --- VISUAL LOG (X-RAY) ---
        if final_contexts:
            logger.info("📝 --- START OF CONTEXT SENT TO LLM ---")
            for i, ctx in enumerate(final_contexts):
                preview = ctx.replace('\n', ' ')[:200]
                logger.info(f"📜 [Chunk {i + 1}]: \"{preview}...\"")
            logger.info("📝 --- END OF CONTEXT ---")

        return final_contexts

    def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """
        Assembles the final prompt and calls the LLM.

        Args:
            question (str): User question.
            contexts (List[str]): Validated contexts.

        Returns:
            str: Generated answer.
        """
        context_block = "\n\n---\n\n".join(contexts)

        user_prompt = f"""
        [RETRIEVED CONTEXTS]
        {context_block}

        [USER QUESTION]
        {question}

        Based strictly on the contexts above, what is the answer?
        """

        try:
            return self.llm.generate_response(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            return "An error occurred while generating the response."

    def _build_empty_response(self, question: str, strategy: Any) -> Dict[str, Any]:
        """Helper to return default response when nothing is found."""
        return {
            "question": question,
            "answer": "The information was not found in the provided documents.",
            "contexts": [],
            "strategy_used": strategy.__class__.__name__
        }
