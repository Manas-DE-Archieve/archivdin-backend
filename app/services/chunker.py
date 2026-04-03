import io
from typing import List

def chunk_text(text: str, size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def extract_pdf_text(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file. 
    If a page is a scanned image (no digital text), uses Tesseract OCR to read it.
    """
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []

    for page in doc:
        # 1. Сначала пытаемся извлечь цифровой текст
        text = page.get_text().strip()

        # 2. Если текста нет или его подозрительно мало (менее 50 символов), 
        # значит перед нами скан/фотография исторического документа
        if len(text) < 50:
            try:
                # Увеличиваем масштаб страницы (zoom) для лучшего качества OCR
                matrix = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=matrix)
                
                # Конвертируем в формат PIL Image
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                
                # Распознаем текст (используем русский и кыргызский языки)
                ocr_text = pytesseract.image_to_string(img, lang="rus+kir")
                text = ocr_text.strip()
            except Exception as e:
                print(f"⚠️ Ошибка OCR на странице: {e}")
                text = ""

        text_parts.append(text)

    return "\n\n".join(text_parts)