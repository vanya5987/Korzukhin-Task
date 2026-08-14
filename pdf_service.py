import io
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

class PdfExtractionError(Exception):
    """Не удалось получить текст из PDF (битый файл, скан без OCR и т.п.)."""

def _normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_text(pdf_bytes: bytes) -> tuple[str, int]:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except PdfReadError as exc:
        raise PdfExtractionError(f"Не удалось прочитать PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise PdfExtractionError(
                "PDF защищён паролем, автоматическая обработка невозможна"
            ) from exc

    pages_text: list[str] = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")

    full_text = _normalize_text("\n".join(pages_text))
    page_count = len(reader.pages)

    if not full_text:
        raise PdfExtractionError(
            "В PDF не найден текстовый слой. Похоже, это скан - "
            "нужна OCR-обработка, которая в данном сервисе не выполняется."
        )

    return full_text, page_count
