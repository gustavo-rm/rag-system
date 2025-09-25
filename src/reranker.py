from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, Dict, Any

class ReRanker:
    """
    Uma classe para reclassificar documentos recuperados usando um modelo Cross-Encoder.
    O Cross-Encoder é mais preciso que a busca vetorial para avaliar a relevância
    de um documento para uma consulta específica.
    """
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Inicializa o ReRanker.

        Parâmetros:
        - model_name (str): O nome de um modelo Cross-Encoder do Hugging Face.
                            'ms-marco-MiniLM-L-6-v2' é um modelo leve e eficaz.
        """
        self.model = CrossEncoder(model_name)
        print(f"ReRanker inicializado com o modelo: {model_name}")

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Reclassifica uma lista de documentos com base em uma consulta.

        Parâmetros:
        - query (str): A pergunta do usuário.
        - documents (List[Dict[str, Any]]): A lista de documentos recuperados do VectorStore.
                                           Espera-se que cada dicionário tenha a chave 'metadata' com o texto.
        - top_n (int): O número de documentos a serem retornados após a reclassificação.

        Retorna:
        - Uma lista de documentos reclassificados e ordenados pela relevância.
        """
        if not documents:
            return []

        # Cria os pares (consulta, texto_do_documento) para o modelo pontuar
        pairs = [[query, doc['metadata']['text']] for doc in documents]

        # Obtém as pontuações de relevância do modelo
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Adiciona a pontuação de relevância a cada documento
        for doc, score in zip(documents, scores):
            doc['relevance_score'] = score

        # Ordena os documentos pela pontuação de relevância (do maior para o menor)
        reranked_docs = sorted(documents, key=lambda x: x['relevance_score'], reverse=True)

        # Retorna os N melhores documentos
        return reranked_docs[:top_n]