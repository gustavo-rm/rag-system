import argparse
import logging
import os

from src.utils.logger import setup_logging

# --- Configuração de Logging ---
setup_logging()
logger = logging.getLogger(__name__)

# Importações dos módulos v3
from src.components.llm import LLM
from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.chunker import Chunker
from src.training.generators import FileTripletGenerator, SyntheticTripletGenerator
from src.training.trainer import EmbeddingTrainer


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Fine-Tuning de Embeddings (RAG v3)")

    # Parâmetros de Entrada
    parser.add_argument('--mode', choices=['file', 'synthetic'], required=True,
                        help="Fonte dos dados: arquivo JSON ou geração sintética via LLM.")
    parser.add_argument('--input', type=str, required=True,
                        help="Caminho do PDF (modo synthetic) ou JSON (modo file).")

    # Parâmetros de Treino
    parser.add_argument('--base_model', type=str, default='paraphrase-multilingual-mpnet-base-v2')
    parser.add_argument('--output_dir', type=str, default='models/finetuned_v3')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_gen', type=int, default=50, help="Qtd de exemplos sintéticos a gerar.")

    args = parser.parse_args()

    # 1. GERAÇÃO DE DADOS
    train_examples = []

    if args.mode == 'file':
        generator = FileTripletGenerator(args.input)
        train_examples = generator.generate()

    elif args.mode == 'synthetic':
        if not os.path.exists(args.input):
            logger.error(f"PDF não encontrado: {args.input}")
            return

        logger.info("Preparando Ingestão para geração sintética...")

        # Pipeline de Ingestão v3
        processor = PDFProcessor(args.input)
        text = processor.extract_text()

        # Usamos chunks menores (384) para treino, pois o modelo MPNet tem limite de 384 tokens
        chunker = Chunker(chunk_size=384, chunk_overlap=0)
        chunks = chunker.chunk_text(text)

        logger.info(f"Texto quebrado em {len(chunks)} trechos.")

        # Carrega LLM Local (Phi-3 é ótimo para gerar perguntas rápidas)
        llm = LLM(method='local', model_name='microsoft/Phi-3-mini-4k-instruct')

        generator = SyntheticTripletGenerator(llm=llm, num_examples=args.num_gen)
        train_examples = generator.generate(chunks=chunks)

    # 2. TREINAMENTO
    if train_examples:
        trainer = EmbeddingTrainer(
            base_model_name=args.base_model,
            batch_size=args.batch_size,
            epochs=args.epochs
        )

        trainer.train(train_examples, output_path=args.output_dir)
    else:
        logger.warning("Nenhum dado gerado. Encerrando.")


if __name__ == "__main__":
    main()