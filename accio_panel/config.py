from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    version: str = os.getenv("ACCIO_VERSION", "0.5.4")
    base_url: str = os.getenv("ACCIO_BASE_URL", "https://phoenix-gw.alibaba.com")
    callback_host: str = os.getenv("ACCIO_CALLBACK_HOST", "127.0.0.1")
    server_host: str = os.getenv(
        "ACCIO_SERVER_HOST",
        os.getenv("ACCIO_CALLBACK_HOST", "127.0.0.1"),
    )
    callback_port: int = int(os.getenv("ACCIO_CALLBACK_PORT", "4097"))
    request_timeout: float = float(os.getenv("ACCIO_REQUEST_TIMEOUT", "15"))
    auto_open_browser: bool = _env_flag("ACCIO_AUTO_OPEN_BROWSER", True)
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("ACCIO_DATA_DIR", str(PROJECT_ROOT / "data"))
        )
    )
    database_url: str = os.getenv("ACCIO_MYSQL", "").strip()
    mysql_host: str = os.getenv("ACCIO_MYSQL_HOST", "").strip()
    mysql_port: int = int(os.getenv("ACCIO_MYSQL_PORT", "3306"))
    mysql_database: str = os.getenv("ACCIO_MYSQL_DATABASE", "").strip()
    mysql_user: str = os.getenv("ACCIO_MYSQL_USER", "").strip()
    mysql_password: str = os.getenv("ACCIO_MYSQL_PASSWORD", "")
    mysql_charset: str = (
        os.getenv("ACCIO_MYSQL_CHARSET", "utf8mb4").strip() or "utf8mb4"
    )

    @property
    def accounts_file(self) -> Path:
        return self.data_dir / "accio-accounts.json"

    @property
    def accounts_dir(self) -> Path:
        return self.data_dir / "accounts"

    @property
    def settings_file(self) -> Path:
        return self.data_dir / "config.json"

    @property
    def stats_file(self) -> Path:
        return self.data_dir / "stats.json"

    @property
    def api_logs_file(self) -> Path:
        return self.data_dir / "api-logs.jsonl"

    @property
    def legacy_settings_file(self) -> Path:
        return self.data_dir / "accio-settings.json"

    @property
    def callback_url(self) -> str:
        return f"http://{self.callback_host}:{self.callback_port}/auth/callback"

    @property
    def database_enabled(self) -> bool:
        if self.database_url:
            return True
        return bool(self.mysql_host and self.mysql_database and self.mysql_user)

    @property
    def storage_backend(self) -> str:
        return "mysql" if self.database_enabled else "file"

    @property
    def database_summary(self) -> str:
        if not self.database_enabled:
            return ""

        if self.database_url:
            parsed = urlparse(self.database_url)
            host = parsed.hostname or ""
            port = parsed.port or 3306
            database = parsed.path.lstrip("/")
            user = unquote(parsed.username or "")
            if host and database:
                account = f"{user}@" if user else ""
                return f"{account}{host}:{port}/{database}"
            return self.database_url

        account = f"{self.mysql_user}@" if self.mysql_user else ""
        return f"{account}{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
