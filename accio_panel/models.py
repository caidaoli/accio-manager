from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_timestamp(value: Any) -> int | None:
    if value in (None, "", 0, "0"):
        return None

    try:
        timestamp = int(float(value))
    except (TypeError, ValueError):
        return None

    if timestamp > 10_000_000_000:
        timestamp //= 1000
    return timestamp


@dataclass(slots=True)
class Account:
    id: str
    name: str
    access_token: str
    refresh_token: str
    utdid: str
    expires_at: int | None = None
    cookie: str | None = None
    manual_enabled: bool = True
    auto_disabled: bool = False
    auto_disabled_reason: str | None = None
    last_quota_check_at: int | None = None
    next_quota_check_at: int | None = None
    next_quota_check_reason: str | None = None
    added_at: str = field(default_factory=now_text)
    updated_at: str = field(default_factory=now_text)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Account":
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or "未命名账号"),
            access_token=str(data.get("accessToken") or ""),
            refresh_token=str(data.get("refreshToken") or ""),
            utdid=str(data.get("utdid") or ""),
            expires_at=normalize_timestamp(data.get("expiresAt")),
            cookie=data.get("cookie"),
            manual_enabled=bool(data.get("manualEnabled", data.get("enabled", True))),
            auto_disabled=bool(data.get("autoDisabled", False)),
            auto_disabled_reason=data.get("autoDisabledReason"),
            last_quota_check_at=normalize_timestamp(data.get("lastQuotaCheckAt")),
            next_quota_check_at=normalize_timestamp(data.get("nextQuotaCheckAt")),
            next_quota_check_reason=data.get("nextQuotaCheckReason"),
            added_at=str(data.get("addedAt") or now_text()),
            updated_at=str(data.get("updatedAt") or data.get("addedAt") or now_text()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "accessToken": self.access_token,
            "refreshToken": self.refresh_token,
            "utdid": self.utdid,
            "expiresAt": self.expires_at,
            "cookie": self.cookie,
            "manualEnabled": self.manual_enabled,
            "autoDisabled": self.auto_disabled,
            "autoDisabledReason": self.auto_disabled_reason,
            "lastQuotaCheckAt": self.last_quota_check_at,
            "nextQuotaCheckAt": self.next_quota_check_at,
            "nextQuotaCheckReason": self.next_quota_check_reason,
            "addedAt": self.added_at,
            "updatedAt": self.updated_at,
        }
