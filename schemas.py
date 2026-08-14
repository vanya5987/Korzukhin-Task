from pydantic import BaseModel, Field

class Penalty(BaseModel):

    condition: str = Field(..., description="За что налагается штраф/неустойка")
    amount: str = Field(
        ...,
        description="Сумма или ставка штрафа как указано в документе "
        "(например, '0.1% от цены контракта за каждый день просрочки')",
    )

class ContractSummary(BaseModel):
    contract_amount: str = Field(
        ..., description="Сумма контракта с валютой, как указано в документе"
    )
    execution_deadline: str = Field(
        ..., description="Сроки выполнения работ/поставки"
    )
    contractor_requirements: list[str] = Field(
        default_factory=list,
        description="Ключевые требования к исполнителю (лицензии, опыт, СРО и т.п.)",
    )
    penalties: list[Penalty] = Field(
        default_factory=list, description="Список штрафов и неустоек"
    )
    summary_note: str | None = Field(
        default=None,
        description="Опционально: важные оговорки модели "
        "(например, если часть данных не найдена в тексте)",
    )


class SummarizeResponse(BaseModel):
    filename: str
    pages_processed: int
    truncated: bool = Field(
        description="True, если текст документа был обрезан из-за лимита на контекст LLM"
    )
    summary: ContractSummary


class ErrorResponse(BaseModel):
    detail: str
