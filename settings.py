from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    API_KEY_NAME: str = os.getenv("API_KEY_NAME", "X-Auth-Token")
    API_KEY: str = os.getenv("API_KEY")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    GOOGLE_CSE_KEY: str = os.getenv("GOOGLE_CSE_KEY")
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID")
    MODEL: str = os.getenv("MODEL", "gpt-4o-mini")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Usage in your main code:
settings = get_settings()