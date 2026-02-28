import logging

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> str:
    """Extract text from a PDF file. Falls back to OCR for scanned pages."""
    doc = fitz.open(file_path)
    pages: list[str] = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(text)
        else:
            # Scanned page — attempt OCR via pytesseract
            text = _ocr_page(page, page_num)
            if text.strip():
                pages.append(text)

    doc.close()
    return "\n\n".join(pages)


def _ocr_page(page: fitz.Page, page_num: int) -> str:
    """OCR a single page using pytesseract."""
    try:
        import pytesseract
        from PIL import Image
        import io

        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        return text
    except ImportError:
        logger.warning(
            "pytesseract/Pillow not installed — skipping OCR for page %d",
            page_num,
        )
        return ""
    except Exception:
        logger.exception("OCR failed for page %d", page_num)
        return ""
