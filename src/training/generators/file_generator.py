import json
import logging
from typing import List
from sentence_transformers.readers import InputExample

from src.training.generators import TripletGenerator

# Logger Configuration
logger = logging.getLogger(__name__)

class FileTripletGenerator(TripletGenerator):
    """
    Loads triplets (Anchor, Positive, Negative) from a JSON file.
    Useful when you already have a manually curated dataset ("Golden Dataset").
    """

    def __init__(self, file_path: str):
        """
        Initializes the generator with the path to the JSON file.

        Args:
            file_path (str): Path to the JSON file containing the dataset.
        """
        self.file_path = file_path

    def generate(self, **kwargs) -> List[InputExample]:
        """
        Reads the file and converts entries into InputExample objects.

        Returns:
            List[InputExample]: List of training examples.
        """
        logger.info(f"📂 Loading triplets from file: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON file: {self.file_path}")
            return []

        examples = []
        for item in data:
            # Basic structure validation
            if all(k in item for k in ('anchor', 'positive', 'negative')):
                examples.append(InputExample(texts=[item['anchor'], item['positive'], item['negative']]))

        logger.info(f"✅ {len(examples)} examples loaded and validated.")
        return examples
