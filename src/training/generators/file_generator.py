import json
from typing import List
from sentence_transformers.readers import InputExample
from .base import TripletGenerator


class FileTripletGenerator(TripletGenerator):
    """Gera tripletos a partir de um arquivo JSON pré-existente."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def generate(self, **kwargs) -> List[InputExample]:
        print(f"Carregando tripletos do arquivo: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em {self.file_path}")
            return []

        examples = [InputExample(texts=[item['anchor'], item['positive'], item['negative']]) for item in data]
        print(f"{len(examples)} exemplos carregados do arquivo.")
        return examples