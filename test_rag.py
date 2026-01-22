"""
Script for testing and evaluating the RAG (Retrieval-Augmented Generation) system components.
It demonstrates how to manually feed data into the evaluator to check metrics like Faithfulness and Answer Relevancy.
"""

from src.evaluation.evaluator import RAGEvaluator

if __name__ == "__main__":

    # 1. Data coming from your RAGSystem.ask()
    rag_output = {
        "question": "Quais são as regras da ABNT para margens?",
        "answer": "As margens devem ser: superior e esquerda 3cm, inferior e direita 2cm.",
        "contexts": [
            "De acordo com a norma 14724, as margens esquerda e superior devem ter 3cm...",
            "A capa deve conter o nome do autor..."
        ]
    }

    # 2. Ground Truth that you (human) created
    ground_truth = "Esquerda e superior 3 cm; direita e inferior 2 cm."

    # 3. Evaluation
    evaluator = RAGEvaluator()  # Gets API KEY from .env

    scores = evaluator.evaluate_single(
        question=rag_output["question"],
        generated_answer=rag_output["answer"],
        retrieved_contexts=rag_output["contexts"],
        ground_truth=ground_truth
    )

    print("Scores:", scores)
    # Expected: Faithfulness ~1.0, Answer Relevancy ~1.0, Context Recall ~1.0
