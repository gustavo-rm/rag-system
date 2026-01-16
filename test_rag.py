from src.evaluation.evaluator import RAGEvaluator

if __name__ == "__main__":

    # 1. Dados vindos do seu RAGSystem.ask()
    rag_output = {
        "question": "Quais são as regras da ABNT para margens?",
        "answer": "As margens devem ser: superior e esquerda 3cm, inferior e direita 2cm.",
        "contexts": [
            "De acordo com a norma 14724, as margens esquerda e superior devem ter 3cm...",
            "A capa deve conter o nome do autor..."
        ]
    }

    # 2. Gabarito que você (humano) criou
    gabarito = "Esquerda e superior 3 cm; direita e inferior 2 cm."

    # 3. Avaliação
    evaluator = RAGEvaluator()  # Pega API KEY do .env

    scores = evaluator.evaluate_single(
        question=rag_output["question"],
        generated_answer=rag_output["answer"],
        retrieved_contexts=rag_output["contexts"],
        ground_truth=gabarito
    )

    print("Scores:", scores)
    # Esperado: Faithfulness ~1.0, Answer Relevancy ~1.0, Context Recall ~1.0