import logging
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingTrainer:
    """
    Gerencia o fine-tuning de modelos de embedding usando a perda MultipleNegativesRankingLoss.

    Esta técnica é eficiente pois usa os outros exemplos do batch como "negativos"
    adicionais, maximizando o aprendizado com menos dados.
    """

    def __init__(self, base_model_name: str, batch_size: int = 16, epochs: int = 1):
        """
        Args:
            base_model_name: Modelo HuggingFace para iniciar (ex: 'sentence-transformers/all-MiniLM-L6-v2').
            batch_size: Tamanho do lote. Maior é melhor para Contrastive Learning, mas exige mais VRAM.
            epochs: Quantas vezes passar pelos dados. 1 a 3 costuma ser suficiente para poucos dados.
        """
        self.base_model_name = base_model_name
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🏋️ Trainer inicializado. Device: {self.device.upper()} | Batch: {batch_size}")

    def train(self, train_examples: List[InputExample], output_path: str):
        """
        Executa o treinamento.

        Args:
            train_examples: Lista de objetos InputExample(texts=[ancora, pos, neg]).
            output_path: Onde salvar o modelo final.
        """
        if not train_examples:
            logger.error("Lista de exemplos vazia. Treinamento abortado.")
            return

        logger.info(f"Carregando modelo base: {self.base_model_name}...")
        model = SentenceTransformer(self.base_model_name, device=self.device)

        # DataLoader prepara os batches
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.batch_size)

        # A Loss function mágica para RAG
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # Warmup de 10% é prática padrão em Transformers
        warmup_steps = int(len(train_dataloader) * self.epochs * 0.1)

        logger.info(f"Iniciando treinamento por {self.epochs} épocas ({len(train_dataloader)} steps/época)...")

        try:
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=self.epochs,
                warmup_steps=warmup_steps,
                output_path=output_path,
                show_progress_bar=True,
                use_amp=True if self.device == 'cuda' else False  # Automatic Mixed Precision (acelera na GPU)
            )
            logger.info(f"🎉 Treinamento concluído com sucesso!")
            logger.info(f"Modelo salvo em: {output_path}")
            logger.info("Dica: Atualize seu 'main.py' para apontar o Embedder para este caminho.")

        except Exception as e:
            logger.critical(f"Falha durante o treinamento: {e}")
            raise e