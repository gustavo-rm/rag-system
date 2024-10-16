from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

nltk.download('punkt')


class Evaluator:
    def __init__(self, metrics=None):
        """
        Inicializa o Evaluator com as métricas desejadas.
        - metrics: lista de métricas que deseja calcular ('bleu', 'rouge').
        """
        self.available_metrics = ['bleu', 'rouge']
        self.metrics = metrics if metrics else ['bleu', 'rouge']

        # Verificar se as métricas solicitadas são válidas
        for metric in self.metrics:
            if metric not in self.available_metrics:
                raise ValueError(f"Métrica {metric} não é suportada. Métricas disponíveis: {self.available_metrics}")

        # Inicializa o RougeScorer se ROUGE for solicitado
        if 'rouge' in self.metrics:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate(self, generated_response, reference_response):
        """
        Avalia a resposta gerada com base nas métricas solicitadas.
        Retorna os scores das métricas escolhidas.
        """
        results = {}

        if 'bleu' in self.metrics:
            bleu_score = self.compute_bleu(generated_response, reference_response)
            results['bleu'] = bleu_score

        if 'rouge' in self.metrics:
            rouge_scores = self.compute_rouge(generated_response, reference_response)
            results.update(rouge_scores)  # Adiciona os resultados ROUGE ao dicionário

        return results

    def compute_bleu(self, generated_response, reference_response):
        """
        Calcula a métrica BLEU para a resposta gerada.
        """
        reference_tokens = [nltk.word_tokenize(reference_response)]
        generated_tokens = nltk.word_tokenize(generated_response)

        # Corrigir a atribuição do smoothing_function
        smoothing_fn = SmoothingFunction().method1

        # Calcular a pontuação BLEU com suavização
        bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing_fn)
        return bleu_score

    def compute_rouge(self, generated_response, reference_response):
        """
        Calcula as métricas ROUGE-1, ROUGE-2, e ROUGE-L.
        """
        scores = self.rouge_scorer.score(reference_response, generated_response)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def add_metric(self, metric_name):
        """
        Adiciona uma nova métrica para ser calculada.
        """
        if metric_name not in self.available_metrics:
            raise ValueError(f"Métrica {metric_name} não é suportada.")

        if metric_name not in self.metrics:
            self.metrics.append(metric_name)

    def remove_metric(self, metric_name):
        """
        Remove uma métrica da lista de métricas a serem calculadas.
        """
        if metric_name in self.metrics:
            self.metrics.remove(metric_name)
