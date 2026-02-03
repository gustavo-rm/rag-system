import logging
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from typing import List

# Logger Configuration
logger = logging.getLogger(__name__)


class EmbeddingTrainer:
    """
    Manages embedding model fine-tuning using MultipleNegativesRankingLoss.

    This technique is efficient because it uses other examples in the batch as additional
    "negatives", maximizing learning with less data.
    """

    def __init__(self, base_model_name: str, batch_size: int = 16, epochs: int = 1):
        """
        Initializes the trainer.

        Args:
            base_model_name (str): HuggingFace model to start with (e.g., 'sentence-transformers/all-MiniLM-L6-v2').
            batch_size (int): Batch size. Larger is better for Contrastive Learning, but requires more VRAM.
            epochs (int): How many times to pass through the data. 1 to 3 is usually sufficient for small datasets.
        """
        self.base_model_name = base_model_name
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🏋️ Trainer initialized. Device: {self.device.upper()} | Batch: {batch_size}")

    def train(self, train_examples: List[InputExample], output_path: str):
        """
        Executes the training loop.

        Args:
            train_examples (List[InputExample]): List of InputExample(texts=[anchor, pos, neg]) objects.
            output_path (str): Where to save the final model.
        """
        if not train_examples:
            logger.error("Empty example list. Training aborted.")
            return

        logger.info(f"Loading base model: {self.base_model_name}...")
        model = SentenceTransformer(self.base_model_name, device=self.device)

        # DataLoader prepares the batches
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.batch_size)

        # The magic Loss function for RAG
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # 10% Warmup is standard practice in Transformers
        warmup_steps = int(len(train_dataloader) * self.epochs * 0.1)

        logger.info(f"Starting training for {self.epochs} epochs ({len(train_dataloader)} steps/epoch)...")

        try:
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=self.epochs,
                warmup_steps=warmup_steps,
                output_path=output_path,
                show_progress_bar=True,
                use_amp=True if self.device == 'cuda' else False  # Automatic Mixed Precision (accelerates on GPU)
            )
            logger.info(f"🎉 Training successfully completed!")
            logger.info(f"Model saved to: {output_path}")
            logger.info("Tip: Update your 'main.py' to point the Embedder to this path.")

        except Exception as e:
            logger.critical(f"Failure during training: {e}")
            raise e
        finally:
            # Explicit cleanup to free VRAM for other components
            if 'model' in locals():
                del model
            if 'train_loss' in locals():
                del train_loss
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                logger.debug("GPU memory cache cleared after training.")
