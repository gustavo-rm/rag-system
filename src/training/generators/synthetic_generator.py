import random
from typing import List
from sentence_transformers.readers import InputExample
from tqdm import tqdm
from .base import TripletGenerator
from src.components.llm import LLM


class SyntheticTripletGenerator(TripletGenerator):
    """Gera tripletos sinteticamente usando um LLM local."""

    def __init__(self, llm: LLM, num_examples: int = 100):
        self.llm = llm
        self.num_examples = num_examples
        self.prompt_template = """
        Com base no trecho de documento abaixo, gere uma pergunta clara e específica que este trecho responde diretamente.
        Retorne APENAS a pergunta, sem nenhum texto adicional.

        [TRECHO DO DOCUMENTO]
        {chunk}
        [/TRECHO DO DOCUMENTO]

        Pergunta:
        """
        self.system_prompt = "Você é um assistente de IA especialista em criar perguntas a partir de textos."

    def generate(self, chunks: List[str], **kwargs) -> List[InputExample]:
        print(f"Iniciando a geração sintética de {self.num_examples} tripletos...")
        if len(chunks) < 2:
            raise ValueError("É necessário pelo menos 2 chunks de texto para gerar tripletos.")

        examples = []
        # Usamos tqdm para visualizar o progresso da geração
        for _ in tqdm(range(self.num_examples), desc="Gerando Tripletos Sintéticos"):
            # Seleciona um chunk para ser o 'positivo' e outro para o 'negativo'
            positive_chunk, negative_chunk = random.sample(chunks, 2)

            # Gera a pergunta (âncora) para o chunk positivo
            prompt = self.prompt_template.format(chunk=positive_chunk)
            anchor_question = self.llm.generate_response(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.7  # Um pouco mais de criatividade na pergunta
            )

            if anchor_question:
                examples.append(InputExample(texts=[anchor_question, positive_chunk, negative_chunk]))

        print(f"{len(examples)} tripletos sintéticos gerados com sucesso.")
        return examples