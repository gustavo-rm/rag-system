import argparse
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
# Importa os componentes necessários do nosso código-fonte em 'src'
from src.llm import LLM
from src.pdf_processor import PDFProcessor
from src.chunker import Chunker
from src.training.generators import FileTripletGenerator, SyntheticTripletGenerator

def main():
    # --- Configuração dos Argumentos da Linha de Comando ---
    parser = argparse.ArgumentParser(description="Script de fine-tuning para modelos de embedding.")
    parser.add_argument('--mode', type=str, required=True, choices=['file', 'synthetic'],
                        help="Modo de geração de dados: 'file' (de um JSON) ou 'synthetic' (com LLM).")
    parser.add_argument('--input_path', type=str, required=True,
                        help="Caminho para o arquivo de entrada (data/train/triplets.json para 'file', data/pdfs/relevo-brasileiro.pdf para 'synthetic').")
    parser.add_argument('--base_model', type=str, default='paraphrase-multilingual-mpnet-base-v2',
                        help="Nome do modelo base do SentenceTransformer para fine-tuning.")
    parser.add_argument('--output_path', type=str, default='models/finetuned-embedder',
                        help="Diretório para salvar o modelo treinado.")
    parser.add_argument('--epochs', type=int, default=4, help="Número de épocas de treinamento.")
    parser.add_argument('--batch_size', type=int, default=16, help="Tamanho do lote de treinamento.")
    parser.add_argument('--num_synthetic_examples', type=int, default=100,
                        help="Número de tripletos a gerar no modo 'synthetic'.")

    args = parser.parse_args()

    # --- Carregamento e Geração de Dados ---
    train_examples = []
    if args.mode == 'file':
        generator = FileTripletGenerator(file_path=args.input_path)
        train_examples = generator.generate()

    elif args.mode == 'synthetic':
        print("Preparando dados para geração sintética...")
        # 1. Processar o PDF para obter os chunks de texto
        processor = PDFProcessor(args.input_path)
        text = processor.extract_text()
        chunker = Chunker(chunk_size=384, chunk_overlap=50)  # Chunks menores são melhores para gerar perguntas
        chunks = chunker.chunk_text(text)

        # 2. Carregar o LLM local para gerar as perguntas
        print("Carregando LLM local para geração de dados...")
        llm = LLM(method='local')  # Usará o Phi-3 quantizado, rápido e eficiente

        # 3. Gerar os tripletos
        generator = SyntheticTripletGenerator(llm=llm, num_examples=args.num_synthetic_examples)
        train_examples = generator.generate(chunks=chunks)

    if not train_examples:
        print("Nenhum dado de treinamento foi gerado. Encerrando.")
        return

    # --- Configuração e Execução do Treinamento ---
    print(f"\nCarregando o modelo base: {args.base_model}")
    model = SentenceTransformer(args.base_model)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)

    print(f"Iniciando fine-tuning por {args.epochs} épocas...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_path,
        show_progress_bar=True
    )

    print(f"\nTreinamento concluído! Modelo salvo em: {args.output_path}")


if __name__ == "__main__":
    main()