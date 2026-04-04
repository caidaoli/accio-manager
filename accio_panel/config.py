from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _compiled_containing_dir() -> Path | None:
    compiled = globals().get("__compiled__")
    containing_dir = getattr(compiled, "containing_dir", "")
    if containing_dir:
        return Path(str(containing_dir))

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return None


def _runtime_root() -> Path:
    return _compiled_containing_dir() or PROJECT_ROOT


def _default_data_dir() -> Path:
    configured = os.getenv("ACCIO_DATA_DIR", "").strip()
    if configured:
        return Path(configured)
    return _runtime_root() / "data"


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
    data_dir: Path = field(default_factory=_default_data_dir)
    database_url: str = os.getenv("ACCIO_MYSQL", "").strip()

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
        return bool(self.database_url)

    @property
    def storage_backend(self) -> str:
        return "mysql" if self.database_enabled else "file"

    @property
    def database_summary(self) -> str:
        if not self.database_url:
            return ""
        return self.database_url
