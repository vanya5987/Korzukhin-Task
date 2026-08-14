from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    llm_timeout_seconds: int = 180
    llm_max_retries: int = 2

    max_pdf_size_mb: int = 25
    max_chars_to_llm: int = 24_000

    @property
    def max_pdf_size_bytes(self) -> int:
        return self.max_pdf_size_mb * 1024 * 1024

@lru_cache
def get_settings() -> Settings:
    return Settings()
