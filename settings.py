from pydantic import BaseSettings

class Settings(BaseSettings):
    LANGUAGE: str
    SUMMARIZER_MODEL: str
    STT_MODEL: str
    STT_PROCESSOR: str

    class Config:
        env_file = ".env"


settings = Settings()