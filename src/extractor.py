import os
import PyPDF2
import fitz  # PyMuPDF


class PDFExtractor:
    def __init__(self, pdf_path):
        """
        Inicializa a classe PDFExtractor com o caminho do arquivo PDF.

        Parâmetros:
        - pdf_path: Caminho do arquivo PDF a ser processado.
        """
        self.pdf_path = pdf_path

    def extract_text(self):
        """
        Extrai todo o texto de um arquivo PDF.

        O método percorre todas as páginas do PDF e extrai o texto de cada uma delas.
        Caso uma página não contenha texto, uma mensagem é exibida.

        Retorna:
        - Uma string contendo todo o texto extraído do PDF.
        """
        text = ""
        with open(self.pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
                else:
                    print(f"A página {page_num + 1} não contém texto.")
        return text.strip()

    def extract_images(self, output_folder='imagens_extraidas'):
        """
        Extrai todas as imagens de um arquivo PDF e as salva em uma pasta de saída.

        Parâmetros:
        - output_folder: Pasta onde as imagens extraídas serão salvas (por padrão, 'imagens_extraidas').

        O método percorre todas as páginas do PDF e extrai qualquer imagem encontrada.
        As imagens são salvas no formato original e nomeadas de acordo com a página e a ordem em que aparecem.

        Retorna:
        - O número total de imagens extraídas do PDF.
        """
        pdf_document = fitz.open(self.pdf_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_count = 0
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            images = page.get_images(full=True)
            if images:
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_filename = os.path.join(output_folder, f"image_{page_number + 1}_{img_index + 1}.{image_ext}")
                    with open(image_filename, "wb") as image_file:
                        image_file.write(image_bytes)
                    print(f"Imagem extraída: {image_filename}")
                    image_count += 1
            else:
                print(f"A página {page_number + 1} não contém imagens.")

        if image_count == 0:
            print("Nenhuma imagem foi encontrada no PDF.")
        return image_count
