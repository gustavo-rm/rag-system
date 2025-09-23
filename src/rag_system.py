# /src/rag_system.py (Arquivo Novo)

from typing import List, Dict, Any

# Importando nossas classes refatoradas
from .pdf_processor import PDFProcessor
from .chunker import Chunker
from .embedder import Embedder
from .stores.base import VectorStore  # Importa a interface, não a implementação!
from .llm import LLM


class RAGSystem:
    def __init__(self, chunker: Chunker, embedder: Embedder, vector_store: VectorStore, llm: LLM):
        """
        Inicializa o sistema RAG com todos os seus componentes.
        (Injeção de Dependência)
        """
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

        # Este é o nosso prompt otimizado, o coração da geração de respostas.
        self.system_prompt = """Você é um assistente especialista e atencioso. Sua tarefa é responder à pergunta do usuário estritamente com base no contexto fornecido.
Regras:
1. Analise o contexto e a pergunta cuidadosamente.
2. Responda de forma concisa e direta, usando apenas as informações encontradas no contexto.
3. Se a resposta não estiver no contexto, responda exatamente: 'A informação não foi encontrada nos documentos fornecidos.'
4. Não adicione nenhuma informação externa ou conhecimento prévio."""

    def setup_pipeline(self, pdf_path: str):
        """
        Executa o pipeline de ingestão de dados: processa um PDF e armazena os embeddings.
        """
        print(f"--- Iniciando pipeline de ingestão para: {pdf_path} ---")

        # 1. Processar o PDF para extrair texto limpo
        processor = PDFProcessor(pdf_path)
        text = processor.extract_text()
        print(f"Texto extraído e limpo. Total de caracteres: {len(text)}")

        # 2. Dividir o texto em chunks semânticos
        chunks = self.chunker.chunk_text(text)
        print(f"Texto dividido em {len(chunks)} chunks.")

        # 3. Gerar embeddings para cada chunk
        embeddings = self.embedder.generate_embeddings(chunks)
        print(f"Embeddings gerados para todos os chunks.")

        # 4. Armazenar os chunks e seus embeddings no Vector Store
        self.vector_store.store_embeddings(chunks, embeddings)
        print("--- Pipeline de ingestão concluído com sucesso! ---")

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Executa o pipeline de consulta: busca por contexto e gera uma resposta.
        """
        print(f"\n--- Nova Pergunta: {question} ---")

        # 1. Gerar o embedding para a pergunta
        query_embedding = self.embedder.generate_embeddings([question])[0]

        # 2. Buscar por chunks relevantes no Vector Store
        search_results = self.vector_store.search(query_embedding, top_k=3)
        retrieved_contexts = [result['metadata']['text'] for result in search_results]
        print(f"Contextos recuperados: {len(retrieved_contexts)}")

        # 3. Construir o prompt para o LLM
        context_str = "\n\n---\n\n".join(retrieved_contexts)
        user_prompt = f"""
[CONTEXTO]
{context_str}
[/CONTEXTO]

Com base estritamente no contexto acima, responda à seguinte pergunta:
Pergunta: {question}
"""
        # 4. Gerar a resposta com o LLM
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
