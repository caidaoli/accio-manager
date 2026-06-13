from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

from .models import now_text


def _truncate(value: Any, limit: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _normalize_log_message(value: Any, *, phase: str, stream: bool) -> str:
    message = str(value or "").strip()
    if phase != "upstream_attempt" and stream and "上游流式请求完成" in message:
        return "流式调用完成"
    return message


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in items:
            items.append(text)
    return items


def _format_bytes(size: int) -> str:
    value = float(max(0, size))
    units = ["B", "KB", "MB", "GB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{int(size)}B"


def _extract_image_summary(payload: dict[str, Any]) -> dict[str, Any]:
    image_blocks = _as_int(
        payload.get("imageBlocks", payload.get("image_blocks")),
        0,
    )
    has_image_data = bool(payload.get("hasImageData", payload.get("has_image_data")))
    if image_blocks > 0:
        has_image_data = True

    image_mime_types = _coerce_string_list(
        payload.get("imageMimeTypes", payload.get("image_mime_types"))
    )
    image_sources = _coerce_string_list(
        payload.get("imageSources", payload.get("image_sources"))
    )
    image_data_chars = _as_int(
        payload.get("imageDataChars", payload.get("image_data_chars")),
        0,
    )
    image_data_bytes = _as_int(
        payload.get("imageDataBytes", payload.get("image_data_bytes")),
        0,
    )

    summary_parts: list[str] = []
    if has_image_data:
        summary_parts.append(f"{max(image_blocks, 1)}块图像")
        if image_mime_types:
            summary_parts.append("/".join(image_mime_types[:2]))
        if image_data_bytes > 0:
            summary_parts.append(_format_bytes(image_data_bytes))
        elif image_data_chars > 0:
            summary_parts.append(f"base64 {image_data_chars}字符")
        elif image_sources:
            summary_parts.append("/".join(image_sources[:2]))

    return {
        "hasImageData": has_image_data,
        "imageBlocks": image_blocks,
        "imageMimeTypes": image_mime_types,
        "imageSources": image_sources,
        "imageDataChars": image_data_chars,
        "imageDataBytes": image_data_bytes,
        "imageSummary": " / ".join(summary_parts) if summary_parts else "-",
    }


MAX_LOG_LINES = 200


class ApiLogStore:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def record(self, payload: dict[str, Any]) -> None:
        image_summary = _extract_image_summary(payload)
        phase = str(payload.get("phase") or "").strip() or "final"
        attempt = _as_int(payload.get("attempt"), 0)
        request_id = str(payload.get("requestId") or "")
        root_request_id = str(payload.get("rootRequestId") or request_id)
        stream = bool(payload.get("stream", True))
        message = _normalize_log_message(
            payload.get("message"),
            phase=phase,
            stream=stream,
        )
        entry = {
            "id": uuid.uuid4().hex,
            "createdAt": now_text(),
            **payload,
            "phase": phase,
            "attempt": attempt,
            "rootRequestId": root_request_id,
            "message": message,
            "hasImageData": payload.get("hasImageData", image_summary["hasImageData"]),
            "imageBlocks": payload.get("imageBlocks", image_summary["imageBlocks"]),
            "imageMimeTypes": payload.get("imageMimeTypes", image_summary["imageMimeTypes"]),
            "imageSources": payload.get("imageSources", image_summary["imageSources"]),
            "imageDataChars": payload.get("imageDataChars", image_summary["imageDataChars"]),
            "imageDataBytes": payload.get("imageDataBytes", image_summary["imageDataBytes"]),
            "imageSummary": payload.get("imageSummary", image_summary["imageSummary"]),
        }
        with self._lock:
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._truncate()

    def _truncate(self) -> None:
        if not self.file_path.exists():
            return
        try:
            lines = self.file_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if len(lines) <= MAX_LOG_LINES:
            return
        kept = lines[-MAX_LOG_LINES:]
        self.file_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    def recent(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            if not self.file_path.exists():
                return []
            try:
                lines = self.file_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return []

        items: list[dict[str, Any]] = []
        for raw_line in reversed(lines):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            image_summary = _extract_image_summary(payload)
            phase = str(payload.get("phase") or "").strip() or "final"
            attempt = _as_int(payload.get("attempt"), 0)
            stream = bool(payload.get("stream", True))
            root_request_id = str(
                payload.get("rootRequestId") or payload.get("requestId") or ""
            )
            message = _normalize_log_message(
                payload.get("message"),
                phase=phase,
                stream=stream,
            )
            detail_payload = {
                **payload,
                "phase": phase,
                "attempt": attempt,
                "rootRequestId": root_request_id,
                "message": message,
            }
            item = {
                "id": str(payload.get("id") or ""),
                "createdAt": str(payload.get("createdAt") or "-"),
                "level": str(payload.get("level") or "info"),
                "event": str(payload.get("event") or "-"),
                "accountName": str(payload.get("accountName") or "-"),
                "accountId": str(payload.get("accountId") or ""),
                "model": str(payload.get("model") or "-"),
                "stream": stream,
                "success": bool(payload.get("success", False)),
                "emptyResponse": bool(payload.get("emptyResponse", False)),
                "stopReason": str(payload.get("stopReason") or "-"),
                "statusCode": str(payload.get("statusCode") or "-"),
                "message": _truncate(message, 160) or "-",
                "inputTokens": int(payload.get("inputTokens") or 0),
                "outputTokens": int(payload.get("outputTokens") or 0),
                "durationMs": int(payload.get("durationMs") or 0),
                "phase": phase,
                "phaseLabel": "上游尝试" if phase == "upstream_attempt" else "最终结果",
                "attempt": attempt,
                "attemptDisplay": str(attempt) if attempt > 0 else "-",
                "requestId": str(payload.get("requestId") or ""),
                "rootRequestId": root_request_id,
                **image_summary,
                "detailJson": json.dumps(detail_payload, ensure_ascii=False, indent=2),
            }
            items.append(item)
            if len(items) >= limit:
                break
        return items
