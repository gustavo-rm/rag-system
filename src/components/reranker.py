import torch
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, Dict, Any
import logging
# Configuração de Logger
logger = logging.getLogger(__name__)


class ReRanker:
    """
    Responsável pelo refinamento da busca (Estágio 2).

    Utiliza um modelo Cross-Encoder (que lê a pergunta e o documento simultaneamente)
    para atribuir uma pontuação de relevância mais precisa do que a busca vetorial simples.
    """

    def __init__(self, model_name: str = 'BAAI/bge-reranker-base', device: str = None):
        """
        Inicializa o modelo de Re-ranking.

        Args:
            model_name (str): Nome do modelo no Hugging Face. 'BAAI/bge-reranker-base' é recomendado para Multilíngue.
            device (str, optional): 'cuda' ou 'cpu'. Se None, detecta automaticamente.
        """
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"🔄 Inicializando ReRanker ({model_name}) em: {self.device}...")
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 3, threshold: float = 0.1) -> List[
        Dict[str, Any]]:
        """
        Reclassifica uma lista de documentos candidatos e filtra os irrelevantes.

        Args:
            query (str): A pergunta do usuário.
            documents (List[Dict]): Documentos recuperados pelo Retriever. Devem conter ['metadata']['text'].
            top_n (int): Número máximo de documentos a retornar.
            threshold (float): Nota de corte (0 a 1). Documentos com relevância abaixo disso são descartados
                               para evitar alucinações baseadas em contexto ruim.

        Returns:
            List[Dict]: Lista ordenada dos melhores documentos com scores normalizados.
        """
        if not documents:
            return []

        # Validação de formato
        valid_docs = [doc for doc in documents if 'metadata' in doc and 'text' in doc['metadata']]
        if not valid_docs:
            return []

        # Prepara pares [Pergunta, Documento] para o modelo
        pairs = [[query, doc['metadata']['text']] for doc in valid_docs]

        # Predição (retorna logits não normalizados)
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Garante formato de array numpy
        if not isinstance(scores, (list, np.ndarray)):
            scores = [scores]

        # Normalização Sigmoide (converte logits -10 a +10 para probabilidade 0 a 1)
        scores_sig = 1 / (1 + np.exp(-np.array(scores)))

        results = []
        for doc, score in zip(valid_docs, scores_sig):
            if score >= threshold:
                doc['relevance_score'] = float(score)
                results.append(doc)

        # Ordena do maior score para o menor
        reranked_docs = sorted(results, key=lambda x: x['relevance_score'], reverse=True)

        return reranked_docs[:top_n]
