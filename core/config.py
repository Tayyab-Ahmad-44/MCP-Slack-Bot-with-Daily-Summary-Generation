
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    VERSION: str = "1.0"
    LOGGING_DIR: str = "logs"
    LLM_MODEL: str = "gpt-4.1-nano"

    SLACK_BOT_TOKEN: str
    SLACK_APP_TOKEN: str
    GROQ_API_KEY: str
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str


    class Config:
        env_file = ".env"



settings = Settings()