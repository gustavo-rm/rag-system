import json
import logging
from typing import List
from sentence_transformers.readers import InputExample

from src.training.generators import TripletGenerator

logger = logging.getLogger(__name__)

class FileTripletGenerator(TripletGenerator):
    """
    Carrega tripletos (Âncora, Positivo, Negativo) de um arquivo JSON.
    Útil quando você já tem um dataset curado manualmente ("Golden Dataset").
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def generate(self, **kwargs) -> List[InputExample]:
        logger.info(f"📂 Carregando tripletos do arquivo: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {self.file_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Arquivo JSON inválido: {self.file_path}")
            return []

        examples = []
        for item in data:
            # Validação básica de estrutura
            if all(k in item for k in ('anchor', 'positive', 'negative')):
                examples.append(InputExample(texts=[item['anchor'], item['positive'], item['negative']]))

        logger.info(f"✅ {len(examples)} exemplos carregados e validados.")
        return examples