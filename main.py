import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config import Settings, get_settings
from llm_service import LlmServiceError, OllamaSummarizer
from pdf_service import PdfExtractionError, extract_text
from schemas import ErrorResponse, SummarizeResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Госзакупки — выжимка контрактов",
    description=(
        "Принимает PDF контракта с сайта госзакупок и возвращает структурированную "
        "выжимку: сумма, сроки, требования к исполнителю, штрафы. "
        "Анализ выполняется локальной моделью через Ollama (llama3.1)."
    ),
    version="1.0.0",
)


@app.get("/health", tags=["service"])
async def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Некорректный файл"},
        422: {"model": ErrorResponse, "description": "Не удалось извлечь текст из PDF"},
        502: {"model": ErrorResponse, "description": "Ошибка при обращении к LLM"},
    },
    tags=["summarize"],
)
async def summarize_contract(
    file: UploadFile = File(..., description="PDF-файл контракта госзакупки"),
) -> SummarizeResponse:
    settings: Settings = get_settings()

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=400,
            detail=f"Ожидается PDF-файл, получен content-type: {file.content_type}",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Загружен пустой файл")

    if len(pdf_bytes) > settings.max_pdf_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Файл слишком большой ({len(pdf_bytes) / 1024 / 1024:.1f} МБ). "
                f"Максимум: {settings.max_pdf_size_mb} МБ."
            ),
        )

    try:
        document_text, page_count = extract_text(pdf_bytes)
    except PdfExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    truncated = len(document_text) > settings.max_chars_to_llm
    if truncated:
        logger.warning(
            "Документ %s обрезан с %d до %d символов перед отправкой в LLM",
            file.filename,
            len(document_text),
            settings.max_chars_to_llm,
        )
        document_text = document_text[: settings.max_chars_to_llm]

    summarizer = OllamaSummarizer(settings)
    try:
        summary = await summarizer.summarize(document_text)
    except LlmServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return SummarizeResponse(
        filename=file.filename or "document.pdf",
        pages_processed=page_count,
        truncated=truncated,
        summary=summary,
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
    logger.exception("Необработанная ошибка при обработке запроса %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервиса"},
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)