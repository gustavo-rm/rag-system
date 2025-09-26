from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from typing import List
from sentence_transformers.readers import InputExample

class EmbeddingTrainer:
    def __init__(self, base_model_name: str, train_examples: List[InputExample], epochs: int, batch_size: int):
        self.base_model_name = base_model_name
        self.train_examples = train_examples
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, output_path: str):
        print(f"\nCarregando o modelo base: {self.base_model_name}")
        model = SentenceTransformer(self.base_model_name)

        train_dataloader = DataLoader(self.train_examples, shuffle=True, batch_size=self.batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        warmup_steps = int(len(train_dataloader) * self.epochs * 0.1)

        print(f"Iniciando fine-tuning por {self.epochs} épocas...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True
        )
        print(f"\nTreinamento concluído! Modelo salvo em: {output_path}")