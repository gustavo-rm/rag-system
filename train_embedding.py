import argparse
import logging
import os
import gc
import torch

from src.utils.logger import setup_logging

# --- Logging Configuration ---
setup_logging()
logger = logging.getLogger(__name__)

# Import v3 modules
from src.components.llm import LLM
from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.chunker import Chunker
from src.training.generators import FileTripletGenerator, SyntheticTripletGenerator
from src.training.trainer import EmbeddingTrainer


def main():
    """
    Main entry point for the Embedding Fine-Tuning Pipeline (RAG v3).

    This script allows fine-tuning an embedding model using either an existing dataset (file mode)
    or synthetic data generated from a PDF document (synthetic mode).

    Usage:
        python train_embedding.py --mode synthetic --input data/pdfs/manual.pdf --num_gen 50 --epochs 3
        python train_embedding.py --mode file --input data/dataset.json --epochs 3
    """
    parser = argparse.ArgumentParser(description="Embedding Fine-Tuning Pipeline (RAG v3)")

    # Input Parameters
    parser.add_argument('--mode', choices=['file', 'synthetic'], required=True,
                        help="Data source: 'file' for JSON dataset or 'synthetic' for LLM generation.")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to PDF (synthetic mode) or JSON (file mode).")

    # Training Parameters
    parser.add_argument('--base_model', type=str, default='paraphrase-multilingual-mpnet-base-v2',
                        help="Base model to fine-tune.")
    parser.add_argument('--output_dir', type=str, default='models/finetuned_v3',
                        help="Directory to save the fine-tuned model.")
    parser.add_argument('--epochs', type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument('--num_gen', type=int, default=50,
                        help="Number of synthetic examples to generate (only for synthetic mode).")

    args = parser.parse_args()

    # 1. DATA GENERATION
    train_examples = []

    if args.mode == 'file':
        generator = FileTripletGenerator(args.input)
        train_examples = generator.generate()

    elif args.mode == 'synthetic':
        if not os.path.exists(args.input):
            logger.error(f"PDF not found: {args.input}")
            return

        logger.info("Preparing Ingestion for synthetic generation...")

        # Ingestion Pipeline v3
        processor = PDFProcessor(args.input)
        text = processor.extract_text()

        # Use smaller chunks (384) for training, as MPNet has a limit of 384 tokens
        chunker = Chunker(chunk_size=384, chunk_overlap=0)
        chunks = chunker.chunk_text(text)

        logger.info(f"Text broken into {len(chunks)} chunks.")

        # Load Local LLM (Phi-3 is great for generating quick questions)
        llm = LLM(method='local', model_name='microsoft/Phi-3-mini-4k-instruct')

        generator = SyntheticTripletGenerator(llm=llm, num_examples=args.num_gen)
        train_examples = generator.generate(chunks=chunks)

    # ==============================================================================
    # 🕵️ MEMORY CORRECTION (Add this block BEFORE checking train_examples)
    # ==============================================================================
    if args.mode == 'synthetic':
        logger.info("🧹 Cleaning LLM from memory to free VRAM for training...")

        # 1. Delete references to heavy objects
        del llm
        del generator

        # 2. Force Python to clean RAM
        gc.collect()

        # 3. Force PyTorch to clean GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"VRAM memory freed. Current allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    # ==============================================================================

    # 2. TRAINING
    if train_examples:
        trainer = EmbeddingTrainer(
            base_model_name=args.base_model,
            batch_size=args.batch_size,
            epochs=args.epochs
        )

        trainer.train(train_examples, output_path=args.output_dir)
    else:
        logger.warning("No data generated. Exiting.")


if __name__ == "__main__":
    main()
