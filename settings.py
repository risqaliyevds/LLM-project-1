from pydantic import BaseSettings

class Settings(BaseSettings):
    LANGUAGE: str
    SUMMARIZER_MODEL: str
    URL: str
    API: str

    class Config:
        env_file = ".env"


settings = Settings()