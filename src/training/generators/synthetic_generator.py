import random
import logging
from typing import List
from tqdm import tqdm
from sentence_transformers.readers import InputExample

# Importações do sistema v3
from src.components.llm import LLM, LLMGenerationError
from src.training.generators import TripletGenerator

logger = logging.getLogger(__name__)


class SyntheticTripletGenerator(TripletGenerator):
    """
    Gera dados de treino sintéticos (GPL - Generative Pseudo Labeling) usando o LLM local.

    Processo:
    1. Pega um chunk de texto (Passagem Positiva).
    2. Pede ao LLM para gerar uma pergunta que aquele chunk responde (Âncora).
    3. Escolhe aleatoriamente outro chunk do documento como 'Negativo' (Hard Negative simplificado).
    """

    def __init__(self, llm: LLM, num_examples: int = 100):
        self.llm = llm
        self.num_examples = num_examples

        self.system_prompt = "Você é um especialista em criar datasets para treinamento de IA."
        self.prompt_template = """
        Abaixo está um trecho de um documento técnico.
        Sua tarefa: Escreva uma pergunta CURTA e OBJETIVA que pode ser respondida EXCLUSIVAMENTE com as informações deste trecho.

        [TRECHO]
        {chunk}
        [/TRECHO]

        Responda APENAS a pergunta. Não adicione "Aqui está a pergunta" ou aspas.
        Pergunta:
        """

    def generate(self, chunks: List[str], **kwargs) -> List[InputExample]:
        """
        Gera os tripletos varrendo a lista de chunks.
        """
        logger.info(f"🧪 Iniciando geração sintética de {self.num_examples} tripletos...")

        if len(chunks) < 2:
            logger.error("Impossível gerar tripletos: Documento possui menos de 2 chunks.")
            return []

        examples = []
        # Tenta gerar até atingir o número desejado ou acabar as tentativas
        attempts = 0
        max_attempts = self.num_examples * 2

        pbar = tqdm(total=self.num_examples, desc="Gerando Dados Sintéticos")

        while len(examples) < self.num_examples and attempts < max_attempts:
            attempts += 1

            # Amostragem de chunks
            positive_chunk, negative_chunk = random.sample(chunks, 2)

            # Gera a pergunta (Âncora)
            try:
                prompt = self.prompt_template.format(chunk=positive_chunk)
                anchor_question = self.llm.generate_response(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    max_new_tokens=60,  # Perguntas costumam ser curtas
                    temperature=0.5,  # Criatividade média para variar o fraseado
                    use_cache=False
                )

                # Validação simples da qualidade da geração
                if len(anchor_question) < 10 or "?" not in anchor_question:
                    continue  # Pula gerações ruins

                examples.append(InputExample(texts=[anchor_question, positive_chunk, negative_chunk]))
                pbar.update(1)

            except LLMGenerationError as e:
                logger.warning(f"Falha na geração do LLM: {e}. Tentando novamente...")
                continue
            except Exception as e:
                logger.error(f"Erro inesperado no loop de geração: {e}")
                break

        pbar.close()
        logger.info(f"✅ Geração concluída. Total de tripletos válidos: {len(examples)}")
        return examples