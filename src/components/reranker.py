import torch
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List
import logging

# Configuração de Logger
logger = logging.getLogger(__name__)


class ReRanker:
    """
    Responsável pelo refinamento da busca (Estágio 2).

    Recebe textos candidatos e utiliza um Cross-Encoder para calcular
    a probabilidade de relevância real em relação à pergunta.
    """

    def __init__(self, model_name: str = 'BAAI/bge-reranker-base', device: str = None):
        """
        Inicializa o modelo de Re-ranking.
        """
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"🔄 Inicializando ReRanker ({model_name}) em: {self.device}...")

        try:
            if self.device == "cuda":
                # Carregamento otimizado para GPU (FP16)
                self.model = CrossEncoder(model_name, device="cpu", automodel_args={"torch_dtype": torch.float16})
                self.model.model.to("cuda")
                logger.info("🚀 ReRanker carregado em CUDA (FP16).")
            else:
                self.model = CrossEncoder(model_name, device="cpu")
                logger.info("✅ ReRanker carregado na CPU.")

        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar na GPU ({e}). Usando CPU.")
            self.model = CrossEncoder(model_name, device="cpu")
            self.device = "cpu"

    def rerank(self, query: str, documents: List[str], top_n: int = 3, threshold: float = 0.01) -> List[str]:
        """
        Reclassifica uma lista de textos.

        Args:
            query (str): A pergunta do usuário.
            documents (List[str]): Lista de TEXTOS puros (strings) vindos do Retriever.
            top_n (int): Quantidade de documentos para manter.
            threshold (float): Score mínimo (0 a 1) para considerar relevante.
                               Baixamos para 0.01 para evitar falsos negativos em BGE.

        Returns:
            List[str]: Lista dos melhores textos ordenados por relevância.
        """
        if not documents:
            logger.warning("ReRanker recebeu lista vazia.")
            return []

        # Remove duplicatas exatas e strings vazias antes de processar
        unique_docs = list(set([doc for doc in documents if doc and doc.strip()]))

        if not unique_docs:
            return []

        # Prepara pares [Pergunta, Texto]
        pairs = [[query, doc] for doc in unique_docs]

        # Predição (Logits brutos: podem ser negativos, ex: -8.5 a +2.1)
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Garante array numpy mesmo se for 1 documento
        if isinstance(scores, float):
            scores = np.array([scores])
        elif isinstance(scores, list):
            scores = np.array(scores)

        # Normalização Sigmoide (Transforma Logits em Probabilidade 0.0 a 1.0)
        # Ex: Logit -2 vira 0.12. Logit 5 vira 0.99.
        scores_sig = 1 / (1 + np.exp(-scores))

        # Combina (Texto, Score)
        scored_results = []
        for doc, score in zip(unique_docs, scores_sig):
            # Log de debug para você ajustar o threshold se precisar
            # logger.debug(f"Score: {score:.4f} | Texto: {doc[:30]}...")

            if score >= threshold:
                scored_results.append((doc, score))

        # Se o threshold filtrou tudo, mas temos documentos, pegamos o "menos pior" (Fallback)
        if not scored_results and unique_docs:
            logger.warning("ReRanker: Threshold muito alto. Retornando o Top-1 'menos pior' como fallback.")
            best_idx = np.argmax(scores_sig)
            return [unique_docs[best_idx]]

        # Ordena decrescente pelo score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Retorna apenas os textos
        final_docs = [doc for doc, score in scored_results[:top_n]]

        return final_docs