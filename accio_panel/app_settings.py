from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_ADMIN_PASSWORD = "admin"
DEFAULT_API_ACCOUNT_STRATEGY = "round_robin"
API_ACCOUNT_STRATEGIES = {"fill", "round_robin"}
PROXY_SCHEMES = {
    "http",
    "https",
    "socks4",
    "socks4a",
    "socks5",
    "socks5h",
}


def normalize_api_account_strategy(value: object) -> str:
    strategy = str(value or DEFAULT_API_ACCOUNT_STRATEGY).strip().lower()
    strategy = strategy.replace("-", "_")
    if strategy not in API_ACCOUNT_STRATEGIES:
        return DEFAULT_API_ACCOUNT_STRATEGY
    return strategy


def normalize_upstream_proxy_url(value: object) -> str:
    normalized = str(value or "").strip().rstrip("/")
    if not normalized:
        return ""

    parsed = urlparse(normalized)
    if parsed.scheme not in PROXY_SCHEMES or not parsed.netloc:
        raise ValueError(
            "代理地址必须以 http://、https://、socks5://、socks5h://、socks4:// 或 socks4a:// 开头"
        )
    return normalized


@dataclass(slots=True)
class PanelSettings:
    upstream_proxy_url: str = ""
    auto_disable_on_empty_quota: bool = True
    auto_enable_on_recovered_quota: bool = True
    api_account_strategy: str = DEFAULT_API_ACCOUNT_STRATEGY
    admin_password: str = DEFAULT_ADMIN_PASSWORD
    session_secret: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PanelSettings":
        return cls(
            upstream_proxy_url=str(data.get("upstreamProxyUrl") or "").strip(),
            auto_disable_on_empty_quota=bool(
                data.get("autoDisableOnEmptyQuota", True)
            ),
            auto_enable_on_recovered_quota=bool(
                data.get("autoEnableOnRecoveredQuota", True)
            ),
            api_account_strategy=normalize_api_account_strategy(
                data.get("apiAccountStrategy", DEFAULT_API_ACCOUNT_STRATEGY)
            ),
            admin_password=str(data.get("adminPassword") or DEFAULT_ADMIN_PASSWORD),
            session_secret=str(data.get("sessionSecret") or ""),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return {
            "upstreamProxyUrl": payload["upstream_proxy_url"],
            "autoDisableOnEmptyQuota": payload["auto_disable_on_empty_quota"],
            "autoEnableOnRecoveredQuota": payload["auto_enable_on_recovered_quota"],
            "apiAccountStrategy": payload["api_account_strategy"],
            "adminPassword": payload["admin_password"],
            "sessionSecret": payload["session_secret"],
        }


def load_panel_settings(payload: dict[str, object] | None) -> tuple[PanelSettings, bool]:
    normalized_payload = payload if isinstance(payload, dict) else {}
    settings = PanelSettings.from_dict(normalized_payload)
    changed = False

    try:
        normalized_proxy = normalize_upstream_proxy_url(
            normalized_payload.get("upstreamProxyUrl")
        )
    except ValueError:
        normalized_proxy = ""
        changed = True
    if settings.upstream_proxy_url != normalized_proxy:
        settings.upstream_proxy_url = normalized_proxy
        changed = True

    raw_strategy = str(
        normalized_payload.get("apiAccountStrategy", DEFAULT_API_ACCOUNT_STRATEGY)
    ).strip().lower()
    raw_strategy = raw_strategy.replace("-", "_")
    if "apiAccountStrategy" not in normalized_payload or (
        normalize_api_account_strategy(normalized_payload.get("apiAccountStrategy"))
        != raw_strategy
    ):
        changed = True

    if not settings.admin_password:
        settings.admin_password = DEFAULT_ADMIN_PASSWORD
        changed = True
    if not settings.session_secret:
        settings.session_secret = secrets.token_urlsafe(32)
        changed = True
    return settings, changed


def normalize_panel_settings(settings: PanelSettings) -> PanelSettings:
    if not settings.admin_password:
        settings.admin_password = DEFAULT_ADMIN_PASSWORD
    settings.upstream_proxy_url = normalize_upstream_proxy_url(
        settings.upstream_proxy_url
    )
    settings.api_account_strategy = normalize_api_account_strategy(
        settings.api_account_strategy
    )
    if not settings.session_secret:
        settings.session_secret = secrets.token_urlsafe(32)
    return settings


class PanelSettingsStore:
    def __init__(self, file_path: Path, legacy_file_path: Path | None = None):
        self.file_path = file_path
        self.legacy_file_path = legacy_file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_payload(self, file_path: Path) -> dict[str, object]:
        raw = file_path.read_text(encoding="utf-8").strip() or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        return payload

    def load(self) -> PanelSettings:
        payload: dict[str, object] | None
        if self.file_path.exists():
            payload = self._load_payload(self.file_path)
        elif self.legacy_file_path and self.legacy_file_path.exists():
            payload = self._load_payload(self.legacy_file_path)
        else:
            payload = None

        settings, changed = load_panel_settings(payload)
        if changed or not self.file_path.exists():
            self.save(settings)
        return settings

    def save(self, settings: PanelSettings) -> PanelSettings:
        settings = normalize_panel_settings(settings)
        self.file_path.write_text(
            json.dumps(settings.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return settings
