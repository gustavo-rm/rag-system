import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict, Any, Optional

# RAGAs - Ferramenta para avaliação de RAG
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Baixar o punkt do NLTK se ainda não foi feito
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:  # Usar LookupError
    print("Baixando o pacote 'punkt' do NLTK...")
    nltk.download('punkt')


class ComprehensiveEvaluator:
    """
    Um avaliador que combina métricas clássicas (BLEU, ROUGE)
    com métricas modernas de avaliação de RAG (via RAGAs).
    """

    def __init__(self):
        """Inicializa os scorers necessários."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # Lista de métricas do RAGAs atualizada
        self.ragas_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        print("ComprehensiveEvaluator inicializado.")

    def _compute_classic_metrics(self, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """Calcula as métricas clássicas que dependem de uma resposta de referência."""
        results = {}

        # --- Cálculo do BLEU ---
        reference_tokens = [nltk.word_tokenize(reference_answer.lower())]
        generated_tokens = nltk.word_tokenize(generated_answer.lower())

        # CORREÇÃO 3: Instanciando a classe primeiro para clareza
        chencherry = SmoothingFunction()
        bleu_score = sentence_bleu(
            reference_tokens,
            generated_tokens,
            smoothing_function=chencherry.method1
        )
        results['bleu'] = bleu_score

        # --- Cálculo do ROUGE ---
        rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
        results['rouge1'] = rouge_scores['rouge1'].fmeasure
        results['rouge2'] = rouge_scores['rouge2'].fmeasure
        results['rougeL'] = rouge_scores['rougeL'].fmeasure

        return results

    def _compute_ragas_metrics(self, question: str, generated_answer: str, contexts: List[str],
                               reference_answer: str) -> Dict[str, float]:
        """Calcula as métricas do RAGAs que avaliam o processo de recuperação e geração."""
        data = {
            "question": [question],
            "answer": [generated_answer],
            "contexts": [contexts],
            "ground_truth": [reference_answer]
        }
        dataset = Dataset.from_dict(data)

        score = evaluate(dataset, metrics=self.ragas_metrics)
        score.pop('dataset', None)  # Remove o objeto do dataset para um resultado mais limpo
        return score

    def evaluate(
            self,
            question: str,
            generated_answer: str,
            retrieved_contexts: List[str],
            reference_answer: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Executa uma avaliação completa do resultado de uma consulta RAG.
        """
        all_results = {}

        if reference_answer:
            classic_scores = self._compute_classic_metrics(generated_answer, reference_answer)
            all_results.update(classic_scores)

            ragas_scores = self._compute_ragas_metrics(question, generated_answer, retrieved_contexts, reference_answer)
            all_results.update(ragas_scores)
        else:
            print("Nenhuma resposta de referência fornecida, pulando métricas clássicas e de recall do RAGAs.")
            # Aqui você poderia rodar RAGAs com métricas que não precisam de `ground_truth`
            # Ex: evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

        return all_results