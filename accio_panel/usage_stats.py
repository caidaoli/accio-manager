from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .models import now_text


def _empty_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "successCalls": 0,
        "failedCalls": 0,
        "inputTokens": 0,
        "outputTokens": 0,
        "lastUsedAt": "",
        "lastStopReason": "",
    }


def _empty_payload() -> dict[str, Any]:
    return {
        "totals": _empty_bucket(),
        "models": {},
        "accounts": {},
        "updatedAt": "",
    }


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class UsageStatsStore:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.file_path.exists():
            return _empty_payload()
        try:
            raw = self.file_path.read_text(encoding="utf-8").strip() or "{}"
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return _empty_payload()
        if not isinstance(payload, dict):
            return _empty_payload()

        totals = payload.get("totals")
        models = payload.get("models")
        accounts = payload.get("accounts")
        if not isinstance(totals, dict):
            totals = _empty_bucket()
        if not isinstance(models, dict):
            models = {}
        if not isinstance(accounts, dict):
            accounts = {}
        return {
            "totals": totals,
            "models": models,
            "accounts": accounts,
            "updatedAt": str(payload.get("updatedAt") or ""),
        }

    def _save_unlocked(self, payload: dict[str, Any]) -> None:
        self.file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def record_message(
        self,
        *,
        account_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        success: bool,
        stop_reason: str | None = None,
    ) -> None:
        with self._lock:
            payload = self._load_unlocked()
            now = now_text()

            totals = payload["totals"]
            model_bucket = payload["models"].setdefault(model, _empty_bucket())
            account_bucket = payload["accounts"].setdefault(account_id, _empty_bucket())

            for bucket in (totals, model_bucket, account_bucket):
                bucket["calls"] = _as_int(bucket.get("calls")) + 1
                if success:
                    bucket["successCalls"] = _as_int(bucket.get("successCalls")) + 1
                else:
                    bucket["failedCalls"] = _as_int(bucket.get("failedCalls")) + 1
                bucket["inputTokens"] = _as_int(bucket.get("inputTokens")) + _as_int(
                    input_tokens
                )
                bucket["outputTokens"] = _as_int(bucket.get("outputTokens")) + _as_int(
                    output_tokens
                )
                bucket["lastUsedAt"] = now
                if stop_reason:
                    bucket["lastStopReason"] = stop_reason

            payload["updatedAt"] = now
            self._save_unlocked(payload)

    def snapshot(self, account_names: dict[str, str]) -> dict[str, Any]:
        with self._lock:
            payload = self._load_unlocked()

        totals = payload["totals"]
        models_raw = payload["models"]
        accounts_raw = payload["accounts"]

        models = sorted(
            (
                {
                    "name": model_name,
                    "calls": _as_int(bucket.get("calls")),
                    "successCalls": _as_int(bucket.get("successCalls")),
                    "failedCalls": _as_int(bucket.get("failedCalls")),
                    "inputTokens": _as_int(bucket.get("inputTokens")),
                    "outputTokens": _as_int(bucket.get("outputTokens")),
                    "lastUsedAt": str(bucket.get("lastUsedAt") or "-"),
                    "lastStopReason": str(bucket.get("lastStopReason") or "-"),
                }
                for model_name, bucket in models_raw.items()
                if isinstance(bucket, dict)
            ),
            key=lambda item: (-item["calls"], item["name"]),
        )

        accounts = sorted(
            (
                {
                    "id": account_id,
                    "name": account_names.get(account_id)
                    or f"已删除账号 {account_id[:8]}",
                    "calls": _as_int(bucket.get("calls")),
                    "successCalls": _as_int(bucket.get("successCalls")),
                    "failedCalls": _as_int(bucket.get("failedCalls")),
                    "inputTokens": _as_int(bucket.get("inputTokens")),
                    "outputTokens": _as_int(bucket.get("outputTokens")),
                    "lastUsedAt": str(bucket.get("lastUsedAt") or "-"),
                    "lastStopReason": str(bucket.get("lastStopReason") or "-"),
                }
                for account_id, bucket in accounts_raw.items()
                if isinstance(bucket, dict)
            ),
            key=lambda item: (-item["calls"], item["name"]),
        )

        return {
            "totals": {
                "calls": _as_int(totals.get("calls")),
                "successCalls": _as_int(totals.get("successCalls")),
                "failedCalls": _as_int(totals.get("failedCalls")),
                "inputTokens": _as_int(totals.get("inputTokens")),
                "outputTokens": _as_int(totals.get("outputTokens")),
                "lastUsedAt": str(totals.get("lastUsedAt") or "-"),
                "lastStopReason": str(totals.get("lastStopReason") or "-"),
            },
            "models": models,
            "accounts": accounts,
            "updatedAt": str(payload.get("updatedAt") or "-"),
        }
