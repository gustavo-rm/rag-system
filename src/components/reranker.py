import torch
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, Dict, Any


class ReRanker:
    """
    Classe para reclassificação de documentos (Reranking).
    Melhorias: Suporte a GPU, Modelos Multilíngues e Filtragem por Score.
    """

    def __init__(self, model_name: str = 'BAAI/bge-reranker-base', device: str = None):
        """
        Inicializa o ReRanker.

        Recomendação de modelos:
        - 'BAAI/bge-reranker-base': Ótimo balanço entre performance e velocidade (Multilíngue).
        - 'BAAI/bge-reranker-large': Melhor precisão, mas mais pesado.
        - 'cross-encoder/ms-marco-MiniLM-L-6-v2': Apenas se o conteúdo for 100% inglês e velocidade for crítica.
        """
        # Detecção automática de device se não for especificado
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🔄 Inicializando ReRanker com modelo: {model_name} no dispositivo: {self.device}...")
        self.model = CrossEncoder(model_name, device=self.device)
        print("✅ ReRanker pronto.")

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 3, threshold: float = 0.1) -> List[
        Dict[str, Any]]:
        """
        Reclassifica e filtra documentos.

        Parâmetros:
        - threshold (float): Pontuação mínima (0 a 1) para considerar um documento relevante. 
                             Ajuda a evitar alucinações removendo "lixo".
        """
        if not documents:
            return []

        # Validação simples para evitar erros de chave
        valid_docs = [doc for doc in documents if 'metadata' in doc and 'text' in doc['metadata']]
        if not valid_docs:
            print("⚠️ Aviso: Nenhum documento com o formato correto ('metadata' -> 'text') encontrado.")
            return []

        # Cria pares
        pairs = [[query, doc['metadata']['text']] for doc in valid_docs]

        # Predição (retorna logits)
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Se scores for um escalar (apenas 1 doc), converte para array
        if not isinstance(scores, (list, np.ndarray)):
            scores = [scores]

        # Normalização Sigmoide (Transforma logits em 0-1 para facilitar leitura)
        # BGE reranker retorna logits, então a sigmoide ajuda a entender a confiança.
        scores_sig = 1 / (1 + np.exp(-np.array(scores)))

        results = []
        for doc, score in zip(valid_docs, scores_sig):
            # Apenas adiciona se passar no corte de qualidade (threshold)
            if score >= threshold:
                doc['relevance_score'] = float(score)  # Garante que é float python nativo
                results.append(doc)

        # Ordena
        reranked_docs = sorted(results, key=lambda x: x['relevance_score'], reverse=True)

        return reranked_docs[:top_n]
