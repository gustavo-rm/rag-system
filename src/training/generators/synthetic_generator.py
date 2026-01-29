import random
import logging
from typing import List
from tqdm import tqdm
from sentence_transformers.readers import InputExample

# v3 System Imports
from src.components.llm import LLM, LLMGenerationError
from src.training.generators import TripletGenerator

# Logger Configuration
logger = logging.getLogger(__name__)


class SyntheticTripletGenerator(TripletGenerator):
    """
    Generates synthetic training data (GPL - Generative Pseudo Labeling) using the local LLM.

    Process:
    1. Takes a text chunk (Positive Passage).
    2. Asks the LLM to generate a question that this chunk answers (Anchor).
    3. Randomly chooses another chunk from the document as 'Negative' (Simplified Hard Negative).
    """

    def __init__(self, llm: LLM, num_examples: int = 100):
        """
        Initializes the synthetic generator.

        Args:
            llm (LLM): The language model used to generate questions.
            num_examples (int): The target number of triplets to generate.
        """
        self.llm = llm
        self.num_examples = num_examples

        self.system_prompt = "You are an expert in creating AI training datasets."
        self.prompt_template = """
        Below is an excerpt from a technical document.
        Your task: Write a SHORT and OBJECTIVE question that can be answered EXCLUSIVELY with the information in this excerpt.

        [EXCERPT]
        {chunk}
        [/EXCERPT]

        Answer ONLY the question. Do not add "Here is the question" or quotes.
        Question:
        """

    def generate(self, chunks: List[str], **kwargs) -> List[InputExample]:
        """
        Generates triplets by scanning the list of chunks.

        Args:
            chunks (List[str]): List of text chunks to serve as positives/negatives.

        Returns:
            List[InputExample]: List of generated training examples.
        """
        logger.info(f"🧪 Starting synthetic generation of {self.num_examples} triplets...")

        if len(chunks) < 2:
            logger.error("Impossible to generate triplets: Document has fewer than 2 chunks.")
            return []

        examples = []
        # Tries to generate until reaching the desired number or running out of attempts
        attempts = 0
        max_attempts = self.num_examples * 2

        pbar = tqdm(total=self.num_examples, desc="Generating Synthetic Data")

        while len(examples) < self.num_examples and attempts < max_attempts:
            attempts += 1
            example = self._generate_single_triplet(chunks)

            if example:
                examples.append(example)
                pbar.update(1)

        pbar.close()
        logger.info(f"✅ Generation completed. Total valid triplets: {len(examples)}")
        return examples

    def _generate_single_triplet(self, chunks: List[str]) -> InputExample:
        """
        Attempts to generate a single valid triplet (Anchor, Positive, Negative).

        Args:
            chunks (List[str]): Available text chunks.

        Returns:
            InputExample: A valid training example or None if generation failed.
        """
        # Chunk sampling
        positive_chunk, negative_chunk = random.sample(chunks, 2)

        # Generate the question (Anchor)
        try:
            prompt = self.prompt_template.format(chunk=positive_chunk)
            anchor_question = self.llm.generate_response(
                prompt=prompt,
                system_prompt=self.system_prompt,
                max_new_tokens=60,  # Questions tend to be short
                temperature=0.5,  # Medium creativity to vary phrasing
                use_cache=False
            )

            # Simple validation of generation quality
            if len(anchor_question) < 10 or "?" not in anchor_question:
                return None # Skip bad generations

            return InputExample(texts=[anchor_question, positive_chunk, negative_chunk])

        except LLMGenerationError as e:
            logger.warning(f"LLM Generation failure: {e}. Retrying...")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in generation: {e}")
            return None
        logger.info(f"✅ Generation completed. Total valid triplets: {len(examples)}")
        return examples
