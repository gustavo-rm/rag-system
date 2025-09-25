from typing import Dict, Any

from .chunker import Chunker
from .embedder import Embedder
from .llm import LLM
from .pdf_processor import PDFProcessor
from .query_transformers import QueryTransformer
from .reranker import ReRanker
from .stores.base import VectorStore


class RAGSystem:
    def __init__(self, chunker: Chunker, embedder: Embedder, vector_store: VectorStore, reranker: ReRanker,
                 llm: LLM, query_transformer: QueryTransformer):
        """
        Inicializa o sistema RAG com todos os seus componentes.
        """
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
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
        # (Este método permanece o mesmo, sem alterações)
        print(f"--- Iniciando pipeline de ingestão para: {pdf_path} ---")
        processor = PDFProcessor(pdf_path)
        text = processor.extract_text()
        print(f"Texto extraído e limpo. Total de caracteres: {len(text)}")
        chunks = self.chunker.chunk_text(text)
        print(f"Texto dividido em {len(chunks)} chunks.")
        embeddings = self.embedder.generate_embeddings(chunks)
        print(f"Embeddings gerados para todos os chunks.")
        self.vector_store.store_embeddings(chunks, embeddings)
        print("--- Pipeline de ingestão concluído com sucesso! ---")

    def ask(self, question: str, retrieval_top_k: int = 20, rerank_top_n: int = 3) -> Dict[str, Any]:
        """
        Executa o pipeline de consulta com transformação de consulta e re-ranking.
        """
        print(f"\n--- Nova Pergunta: {question} ---")

        # 1. Etapa de Transformação de Consulta
        transformed_queries = self.query_transformer.transform(question)

        # 2. Gerar embeddings para a(s) consulta(s) transformada(s)
        query_embeddings = self.embedder.generate_embeddings(transformed_queries)

        # 3. Etapa de Recuperação: Buscar com todos os embeddings e unir os resultados
        print(f"Recuperando os {retrieval_top_k} documentos candidatos...")
        all_candidate_docs = []
        for embedding in query_embeddings:
            all_candidate_docs.extend(self.vector_store.search(embedding, top_k=retrieval_top_k))

        # Desduplicar os resultados, mantendo o de maior score se houver sobreposição
        unique_docs_dict = {doc['id']: doc for doc in
                            sorted(all_candidate_docs, key=lambda x: x.get('score', 0), reverse=True)}
        candidate_docs = list(unique_docs_dict.values())

        # 4. Etapa de Re-ranking: Usar o Cross-Encoder para reclassificar
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