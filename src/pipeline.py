from typing import Dict, Any, List

from src.ingestion.chunker import Chunker
from src.components.embedder import Embedder
from src.components.llm import LLM
from src.ingestion.pdf_processor import PDFProcessor
from .query_transformers import QueryTransformer
from src.components.reranker import ReRanker
from src.components.hybrid_retriever import HybridRetriever


class RAGSystem:
    def __init__(self, chunker: Chunker, embedder: Embedder, retriever: HybridRetriever, reranker: ReRanker,
                 llm: LLM, query_transformer: QueryTransformer):
        """
        Inicializa o sistema RAG com todos os seus componentes.

        Args:
            retriever (HybridRetriever): O orquestrador de busca (BM25 + VectorStore).
                                         Substitui a antiga injeção direta de 'vector_store'.
        """
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.query_transformer = query_transformer

        self.system_prompt = """Você é um assistente especialista e atencioso. Sua tarefa é responder à pergunta do usuário estritamente com base no contexto fornecido.
                            Regras:
                            1. Analise o contexto e a pergunta cuidadosamente.
                            2. Responda de forma concisa e direta, usando apenas as informações encontradas no contexto.
                            3. Se a resposta não estiver no contexto, responda exatamente: 'A informação não foi encontrada nos documentos fornecidos.'
                            4. Não adicione nenhuma informação externa ou conhecimento prévio."""

    def setup_pipeline(self, pdf_path: str):
        print(f"--- Iniciando pipeline de ingestão para: {pdf_path} ---")
        processor = PDFProcessor(pdf_path)
        text = processor.extract_text()
        print(f"Texto extraído e limpo. Total de caracteres: {len(text)}")

        chunks = self.chunker.chunk_text(text)
        print(f"Texto dividido em {len(chunks)} chunks.")

        embeddings = self.embedder.generate_embeddings(chunks)
        print(f"Embeddings gerados para todos os chunks.")

        # Usamos o HybridRetriever para adicionar documentos.
        # Isso garante que ele salve no Banco Vetorial E atualize o índice BM25 em memória.
        self.retriever.add_documents(chunks, embeddings)

        print("--- Pipeline de ingestão concluído com sucesso! ---")

    def ask(self, question: str, retrieval_top_k: int = 20, rerank_top_n: int = 3) -> Dict[str, Any]:
        """
        Executa o pipeline de consulta com transformação de consulta, busca híbrida e re-ranking.
        """
        print(f"\n--- Nova Pergunta: {question} ---")

        # 1. Etapa de Transformação de Consulta (Multi-Query / HyDE)
        transformed_queries = self.query_transformer.transform(question)

        # 2. Gerar embeddings para as consultas transformadas
        query_embeddings = self.embedder.generate_embeddings(transformed_queries)

        # 3. Etapa de Recuperação Híbrida
        print(f"Recuperando candidatos (Híbrido) para {len(transformed_queries)} variações de query...")
        all_candidate_docs = []

        # Precisamos do TEXTO (para BM25) e do VETOR (para Embeddings)
        for query_text, query_vec in zip(transformed_queries, query_embeddings):
            results = self.retriever.search(
                query_text=query_text,  # Vai para o BM25
                query_embedding=query_vec,  # Vai para o Chroma/Pinecone
                top_k=retrieval_top_k
            )
            all_candidate_docs.extend(results)

        # DEBUG
        print(f"INFO: Total de documentos brutos recuperados (antes da desduplicação): {len(all_candidate_docs)}")

        # Desduplicar os resultados, mantendo o de maior score se houver sobreposição
        # Nota: O score do BM25 e do Vetor tem escalas diferentes, mas o ReRanker resolve isso.
        unique_docs_dict = {}
        for doc in all_candidate_docs:
            doc_id = doc['id']
            # Se o documento ainda não foi adicionado ou se o novo tem score maior (ex: match exato de vetor)
            if doc_id not in unique_docs_dict:
                unique_docs_dict[doc_id] = doc
            else:
                # Lógica: manter o que tiver maior score para priorizar
                if doc.get('score', 0) > unique_docs_dict[doc_id].get('score', 0):
                    unique_docs_dict[doc_id] = doc

        candidate_docs = list(unique_docs_dict.values())

        # 4. Etapa de Re-ranking: O Cross-Encoder decide quem realmente é relevante
        print(f"Reclassificando {len(candidate_docs)} documentos para encontrar os {rerank_top_n} melhores...")
        reranked_docs = self.reranker.rerank(question, candidate_docs, top_n=rerank_top_n)

        retrieved_contexts = [doc['metadata']['text'] for doc in reranked_docs]
        print(f"Contextos finais selecionados após re-ranking: {len(retrieved_contexts)}")

        # 5. Construir o prompt para o LLM
        context_str = "\n\n---\n\n".join(retrieved_contexts)
        user_prompt = f"""
        [CONTEXTO]
        {context_str}
        [/CONTEXTO]

        Com base estritamente no contexto acima, responda à seguinte pergunta:
        Pergunta: {question}
        """

        # 6. Gerar a resposta com o LLM
        answer = self.llm.generate_response(
            prompt=user_prompt,
            system_prompt=self.system_prompt
        )
        print(f"Resposta Gerada: {answer}")

        return {
            "question": question,
            "answer": answer,
            "contexts": retrieved_contexts
        }