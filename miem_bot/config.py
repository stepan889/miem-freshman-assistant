import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CATALOG_URL = "https://www.hse.ru/ba/isct/courses?course=1.1.1.4&page=1&year=2026"
ACADEMIC_YEAR = "2026/2027"
DATA_FILE = ROOT / "data" / "courses.json"


@dataclass
class Settings:
    telegram_token: str = ""
    api_key: str = ""
    base_url: str = "https://api.agentapi.ru/v1/ai"
    model: str = "deepseek/deepseek-v4-flash"
    allowed_users: set[int] = field(default_factory=set)
    concurrency: int = 3
    cooldown: float = 3

    @classmethod
    def load(cls):
        load_dotenv(ROOT / ".env")
        settings = cls(
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            api_key=os.getenv("AGENTAPI_KEY", "").strip(),
            base_url=os.getenv("AGENTAPI_BASE_URL", cls.base_url).rstrip("/"),
            model=os.getenv("AGENTAPI_MODEL", cls.model),
            allowed_users={int(x.strip()) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()},
            concurrency=int(os.getenv("MAX_CONCURRENT_REQUESTS", "3")),
            cooldown=float(os.getenv("REQUEST_COOLDOWN_SECONDS", "3")),
        )
        if settings.concurrency < 1 or settings.cooldown < 0:
            raise ValueError("Число параллельных запросов должно быть >= 1, интервал >= 0.")
        if not settings.base_url.startswith("https://"):
            raise ValueError("AGENTAPI_BASE_URL должен использовать HTTPS.")
        return settings

    def require_keys(self, telegram=False):
        missing = []
        if not self.api_key:
            missing.append("AGENTAPI_KEY")
        if telegram and not self.telegram_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if missing:
            raise ValueError("Заполните в .env: " + ", ".join(missing))
