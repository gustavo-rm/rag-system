import fitz  # PyMuPDF
import os
import re
import logging
from src.utils.exceptions import IngestionError

# Logger Configuration
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Processes PDF files to extract and clean text, as well as extract images.
    Uses the PyMuPDF (fitz) library for superior performance and accuracy.
    """

    def __init__(self, pdf_path: str):
        """
        Initializes the processor with the PDF file path.

        Args:
            pdf_path (str): Path of the PDF file to be processed.

        Raises:
            FileNotFoundError: If the PDF file is not found.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self.pdf_path = pdf_path

    def _clean_text(self, text: str) -> str:
        """
        Performs basic cleaning on extracted text to improve quality.
        - Joins words broken by hyphens.
        - Removes excessive line breaks.
        - Normalizes whitespace.

        Args:
            text (str): The raw text to clean.

        Returns:
            str: The cleaned text.
        """
        # 1. Joins words separated by a hyphen at the end of the line
        # Ex: "intelli- gence" -> "intelligence"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # 2. Replaces multiple spaces or line breaks with a single space
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def extract_text(self, clean: bool = True) -> str:
        """
        Extracts all text from a PDF file using PyMuPDF for high fidelity.

        Args:
            clean (bool): If True, applies the cleaning function to the extracted text.

        Returns:
            str: A string containing all text extracted from the PDF.

        Raises:
            IngestionError: If PDF processing fails.
        """
        full_text = ""
        try:
            with fitz.open(self.pdf_path) as doc:
                for page in doc:
                    full_text += page.get_text() + " "
        except Exception as e:
            logger.error(f"Error processing PDF {self.pdf_path}: {e}")
            raise IngestionError(f"Failed to extract text from {self.pdf_path}: {e}", e)

        if clean:
            return self._clean_text(full_text)

        return full_text.strip()

    def extract_images(self, output_folder: str = 'extracted_images'):
        """
        Extracts all images from a PDF file.

        Args:
            output_folder (str): Folder to save extracted images.

        Returns:
            int: The number of images extracted.

        Raises:
            IngestionError: If image extraction or saving fails.
        """
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            image_count = 0
            with fitz.open(self.pdf_path) as doc:
                for page_num in range(len(doc)):
                    for img_index, img in enumerate(doc.get_page_images(page_num)):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image_filename = os.path.join(output_folder, f"image_{page_num + 1}_{img_index + 1}.{image_ext}")

                        with open(image_filename, "wb") as image_file:
                            image_file.write(image_bytes)
                        image_count += 1

            logger.info(f"Total of {image_count} images extracted to '{output_folder}'.")
            return image_count
        except Exception as e:
            logger.error(f"Error extracting images from PDF {self.pdf_path}: {e}")
            raise IngestionError(f"Failed to extract images from {self.pdf_path}: {e}", e)


# --- Usage Example ---
if __name__ == '__main__':
    # Create a sample PDF file called 'example.pdf' to test
    # or point to an existing PDF.
    pdf_file_path = 'example.pdf'  # Replace with your PDF path
    if not os.path.exists(pdf_file_path):
        logger.warning(f"Sample file '{pdf_file_path}' not found. Create one to test the code.")
    else:
        processor = PDFProcessor(pdf_path=pdf_file_path)

        # Extract and clean text
        logger.info("--- Extracting Clean Text ---")
        cleaned_text = processor.extract_text()
        logger.info(cleaned_text[:1000] + "...")  # Prints the first 1000 characters

        logger.info("\n" + "=" * 50 + "\n")

        # Extract images
        logger.info("--- Extracting Images ---")
        processor.extract_images()
