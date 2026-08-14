import json
import logging

import httpx
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import Settings
from schemas import ContractSummary

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Ты - юридический ассистент, который анализирует документы государственных \
закупок (контракты, техзадания, извещения). Твоя задача — извлечь из текста \
документа только фактические данные, ничего не придумывая.

Верни ответ СТРОГО в виде JSON со следующими полями:
- contract_amount: сумма контракта с валютой (строка). Если в тексте есть \
несколько сумм (НМЦК, цена контракта) - укажи именно цену/сумму контракта.
- execution_deadline: сроки выполнения работ или поставки (строка).
- contractor_requirements: список ключевых требований к исполнителю \
(лицензии, допуски СРО, опыт работы, квалификация персонала и т.п.).
- penalties: список объектов {"condition": "...", "amount": "..."} - \
за что и какой штраф/пеня/неустойка предусмотрены.
- summary_note: если какие-то из полей не найдены в тексте — кратко укажи \
это здесь (не выдумывай данные, которых нет в документе).

Если какое-то поле невозможно определить из текста — используй значение \
"не указано в документе", но не оставляй поле пустым и не придумывай данные.
Отвечай только JSON, без пояснений и markdown-разметки вокруг него.
"""

class LlmServiceError(Exception):
    """Ошибка при обращении к LLM или разборе её ответа."""

def _build_user_prompt(document_text: str) -> str:
    return f"Текст документа госзакупки:\n\n{document_text}"

def _extract_json_payload(raw_content: str) -> dict:
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    return json.loads(cleaned)

class OllamaSummarizer:
    def __init__(self, settings: Settings):
        self._settings = settings

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.HTTPError, LlmServiceError)),
    )
    async def _call_ollama(self, prompt: str) -> str:
        url = f"{self._settings.ollama_base_url}/api/chat"
        payload = {
            "model": self._settings.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        }

        async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
            except httpx.ConnectError as exc:
                raise LlmServiceError(
                    f"Не удалось подключиться к Ollama по адресу {url}. "
                    "Убедитесь, что сервер Ollama запущен (`ollama serve`) "
                    f"и модель загружена (`ollama pull {self._settings.ollama_model}`)."
                ) from exc
            except httpx.HTTPStatusError as exc:
                raise LlmServiceError(
                    f"Ollama вернула ошибку {exc.response.status_code}: {exc.response.text}"
                ) from exc

        data = response.json()
        content = data.get("message", {}).get("content")
        if not content:
            raise LlmServiceError("Ollama вернула пустой ответ")
        return content

    async def summarize(self, document_text: str) -> ContractSummary:
        prompt = _build_user_prompt(document_text)
        raw_content = await self._call_ollama(prompt)

        try:
            payload = _extract_json_payload(raw_content)
        except json.JSONDecodeError as exc:
            logger.error("LLM вернула невалидный JSON: %s", raw_content)
            raise LlmServiceError(
                "Модель вернула невалидный JSON, не удалось разобрать ответ"
            ) from exc

        try:
            return ContractSummary.model_validate(payload)
        except ValidationError as exc:
            logger.error("Ответ LLM не прошёл валидацию схемы: %s", payload)
            raise LlmServiceError(
                f"Ответ модели не соответствует ожидаемой структуре: {exc}"
            ) from exc