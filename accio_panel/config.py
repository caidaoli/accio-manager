from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    version: str = os.getenv("ACCIO_VERSION", "0.5.2")
    base_url: str = os.getenv("ACCIO_BASE_URL", "https://phoenix-gw.alibaba.com")
    callback_host: str = os.getenv("ACCIO_CALLBACK_HOST", "127.0.0.1")
    callback_port: int = int(os.getenv("ACCIO_CALLBACK_PORT", "4097"))
    request_timeout: float = float(os.getenv("ACCIO_REQUEST_TIMEOUT", "15"))
    auto_open_browser: bool = _env_flag("ACCIO_AUTO_OPEN_BROWSER", True)
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("ACCIO_DATA_DIR", str(PROJECT_ROOT / "data"))
        )
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
    def legacy_settings_file(self) -> Path:
        return self.data_dir / "accio-settings.json"

    @property
    def callback_url(self) -> str:
        return f"http://{self.callback_host}:{self.callback_port}/auth/callback"
