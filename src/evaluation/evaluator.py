import logging
import os
import pandas as pd
from typing import List, Dict, Optional, Any
from datasets import Dataset

# Ragas Metrics
from ragas import evaluate
from ragas.metrics import (
    faithfulness,  # O modelo alucinou? (Geração)
    answer_relevancy,  # Respondeu o que foi perguntado? (Geração)
    context_precision,  # O Retriever trouxe documentos úteis no topo? (Recuperação)
    context_recall,  # O Retriever trouxe TODA a informação necessária? (Recuperação)
)
# Integração com LangChain (Necessário para o RAGAS funcionar bem)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configuração de Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Avaliador Semântico para sistemas RAG (Retrieval-Augmented Generation).

    Usa métricas baseadas em LLM, que avaliam o significado e a veracidade das respostas.

    Esta classe atua como um 'Juiz' independente. Recomenda-se usar um modelo
    forte (ex: GPT-4o) para a avaliação, independente do modelo usado no RAG.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Inicializa o avaliador configurando o LLM 'Juiz'.

        Args:
            openai_api_key (str): Chave da OpenAI. Se None, tenta pegar do ambiente.
                                  O RAGAS funciona melhor com a OpenAI como juiz.
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning(
                "⚠️ RAGEvaluator: API Key da OpenAI não encontrada. A avaliação pode falhar se não houver configuração global.")

        # Configura o LLM e Embeddings especificamente para o RAGAS (Juiz)
        # Usamos gpt-4o-mini ou gpt-4 para avaliação por serem mais rigorosos
        self.judge_llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.api_key)
        self.judge_embeddings = OpenAIEmbeddings(api_key=self.api_key)

        # Métricas divididas por caso de uso
        self.metrics_with_ground_truth = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

        self.metrics_no_ground_truth = [
            faithfulness,
            answer_relevancy
        ]

        logger.info("⚖️ RAGEvaluator (v3.1) inicializado com Juiz OpenAI.")

    def evaluate_single(self,
                        question: str,
                        generated_answer: str,
                        retrieved_contexts: List[str],
                        ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Avalia uma única interação do RAG.

        Args:
            question (str): A pergunta do usuário.
            generated_answer (str): A resposta final do Chatbot.
            retrieved_contexts (List[str]): Lista de textos recuperados (contexto).
            ground_truth (str, optional): A resposta ideal (gabarito).

        Returns:
            Dict[str, float]: Dicionário com as pontuações (0.0 a 1.0).
        """
        # Prepara os dados no formato que o RAGAS espera
        data = {
            "question": [question],
            "answer": [generated_answer],
            "contexts": [retrieved_contexts],
        }

        metrics_to_use = self.metrics_no_ground_truth

        if ground_truth:
            data["ground_truth"] = [ground_truth]
            metrics_to_use = self.metrics_with_ground_truth
            logger.info("Executando avaliação completa (com Ground Truth)...")
        else:
            logger.info("Executando avaliação parcial (sem Ground Truth - Apenas Geração)...")

        try:
            dataset = Dataset.from_dict(data)

            # Executa a avaliação
            # Passamos o llm e embeddings explicitamente para garantir que o RAGAS use nossa config
            results = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
                llm=self.judge_llm,
                embeddings=self.judge_embeddings,
                raise_exceptions=False
            )

            # Converte para dicionário Python padrão
            # O objeto results do RAGAS se comporta como dict
            scores = {k: round(v, 4) for k, v in results.items()}

            logger.info(f"📊 Resultados da Avaliação: {scores}")
            return scores

        except Exception as e:
            logger.error(f"❌ Falha ao executar RAGAS: {e}")
            return {"error": 0.0}

    def evaluate_batch(self, samples: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Avalia um lote de perguntas (Golden Dataset) e retorna um DataFrame.

        Args:
            samples (List[Dict]): Lista de dicts contendo chaves:
                                  'question', 'answer', 'contexts', 'ground_truth' (opcional).

        Returns:
            pd.DataFrame: Tabela com resultados comparativos.
        """
        logger.info(f"Iniciando avaliação em lote de {len(samples)} itens...")

        # Transforma lista de dicts em dict de listas (formato columnar do Dataset)
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        has_gt = all("ground_truth" in s for s in samples)

        for s in samples:
            data["question"].append(s["question"])
            data["answer"].append(s["answer"])
            data["contexts"].append(s["contexts"])
            if has_gt:
                data["ground_truth"].append(s.get("ground_truth", ""))

        if not has_gt:
            del data["ground_truth"]
            metrics = self.metrics_no_ground_truth
        else:
            metrics = self.metrics_with_ground_truth

        dataset = Dataset.from_dict(data)

        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.judge_llm,
            embeddings=self.judge_embeddings
        )

        df = results.to_pandas()
        logger.info("Avaliação em lote concluída.")
        return df