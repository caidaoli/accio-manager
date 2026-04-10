from __future__ import annotations

import json
import time
from typing import Any, Awaitable, Callable, Iterator, TypeVar

import requests

from .anthropic_proxy import UpstreamTurnError
from .api_logs import ApiLogStore
from .gemini_proxy import summarize_gemini_response
from .models import Account


_ResponseT = TypeVar("_ResponseT")


def upstream_turn_error_message(exc: UpstreamTurnError) -> str:
    message = exc.error_message or "上游返回错误。"
    if exc.error_code:
        return f"上游返回错误 [{exc.error_code}]: {message}"
    return f"上游返回错误: {message}"


def should_retry_upstream_turn_error(exc: UpstreamTurnError) -> bool:
    return str(exc.error_code or "").strip() in {"555"}


def record_proxy_log(
    api_log_store: ApiLogStore,
    *,
    event: str,
    model: str,
    stream: bool,
    strategy: str,
    request_id: str,
    success: bool,
    stop_reason: str,
    message: str,
    status_code: int,
    account: Account | None = None,
    quota: dict[str, Any] | None = None,
    empty_response: bool = False,
    messages_count: int = 0,
    max_tokens: int | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration_ms: int = 0,
    level: str | None = None,
    phase: str | None = None,
    attempt: int | None = None,
    root_request_id: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    resolved_level = level or ("info" if success else "error")
    if empty_response and level is None:
        resolved_level = "warn"

    payload: dict[str, Any] = {
        "level": resolved_level,
        "event": event,
        "success": success,
        "emptyResponse": empty_response,
        "accountId": account.id if account else "",
        "accountName": account.name if account else "-",
        "fillPriority": account.fill_priority if account else None,
        "model": model,
        "stream": stream,
        "strategy": strategy,
        "requestId": request_id,
        "message": str(message or "").strip(),
        "statusCode": status_code,
        "stopReason": stop_reason,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "remainingQuota": quota.get("remaining_value") if isinstance(quota, dict) else None,
        "usedQuota": quota.get("used_value") if isinstance(quota, dict) else None,
        "messagesCount": messages_count,
        "durationMs": duration_ms,
    }
    if phase:
        payload["phase"] = phase
    if attempt is not None:
        payload["attempt"] = attempt
    if root_request_id:
        payload["rootRequestId"] = root_request_id
    if max_tokens is not None:
        payload["maxTokens"] = max_tokens
    if extra_fields:
        for key, value in extra_fields.items():
            if value is not None:
                payload[key] = value
    api_log_store.record(payload)


def make_upstream_attempt_logger(
    api_log_store: ApiLogStore,
    *,
    event: str,
    model: str,
    strategy: str,
    root_request_id: str,
    messages_count: int = 0,
    max_tokens: int | None = None,
) -> Callable[..., None]:
    def _record_attempt(
        account: Account,
        quota: dict[str, Any],
        request_id: str,
        *,
        attempt: int,
        stream: bool,
        success: bool,
        stop_reason: str,
        message: str,
        status_code: int,
        duration_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        empty_response: bool = False,
        extra_fields: dict[str, Any] | None = None,
        level: str | None = None,
    ) -> None:
        record_proxy_log(
            api_log_store,
            event=event,
            model=model,
            stream=stream,
            strategy=strategy,
            request_id=request_id,
            success=success,
            stop_reason=stop_reason,
            message=message,
            status_code=status_code,
            account=account,
            quota=quota,
            empty_response=empty_response,
            messages_count=messages_count,
            max_tokens=max_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            level=level,
            phase="upstream_attempt",
            attempt=attempt,
            root_request_id=root_request_id or request_id,
            extra_fields=extra_fields,
        )

    return _record_attempt


async def request_upstream_or_error(
    request_call: Callable[[], Awaitable[requests.Response]],
    *,
    account: Account,
    quota: dict[str, Any],
    request_id: str,
    attempt: int,
    stream: bool,
    started_at: float,
    record_attempt: Callable[..., None],
    build_error_response: Callable[[int, str, str], _ResponseT],
    usage_failure_recorder: Callable[[str], None] | None = None,
    retry_reason: str | None = None,
) -> requests.Response | _ResponseT:
    extra_fields = {"retryReason": retry_reason} if retry_reason else None
    try:
        upstream_response = await request_call()
    except requests.RequestException as exc:
        if usage_failure_recorder is not None:
            usage_failure_recorder("request_exception")
        record_attempt(
            account,
            quota,
            request_id,
            attempt=attempt,
            stream=stream,
            success=False,
            stop_reason="request_exception",
            message=f"上游请求失败: {exc}",
            status_code=502,
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            extra_fields=extra_fields,
        )
        return build_error_response(502, f"上游请求失败: {exc}", "request_exception")

    if upstream_response.ok:
        return upstream_response

    upstream_text = ""
    try:
        upstream_text = upstream_response.text[:500]
    finally:
        upstream_response.close()
    if usage_failure_recorder is not None:
        usage_failure_recorder("upstream_error")
    record_attempt(
        account,
        quota,
        request_id,
        attempt=attempt,
        stream=stream,
        success=False,
        stop_reason="upstream_error",
        message=upstream_text or "上游返回错误。",
        status_code=upstream_response.status_code,
        duration_ms=int((time.perf_counter() - started_at) * 1000),
        extra_fields=extra_fields,
    )
    return build_error_response(
        upstream_response.status_code,
        upstream_text or "上游返回错误。",
        "upstream_error",
    )


def is_stream_summary_empty(summary: dict[str, Any]) -> bool:
    return (
        int(summary.get("text_chars") or 0) <= 0
        and int(summary.get("tool_use_blocks") or 0) <= 0
    )


def summarize_non_stream_payload(payload: dict[str, Any]) -> dict[str, int | bool]:
    content = payload.get("content")
    if not isinstance(content, list):
        return {"text_chars": 0, "tool_use_blocks": 0, "empty_response": True}

    text_chars = 0
    tool_use_blocks = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type == "text":
            text_chars += len(str(block.get("text") or ""))
        elif block_type == "tool_use":
            tool_use_blocks += 1

    return {
        "text_chars": text_chars,
        "tool_use_blocks": tool_use_blocks,
        "empty_response": text_chars <= 0 and tool_use_blocks <= 0,
    }


def parse_sse_chunk_payloads(chunk: bytes | str) -> list[dict[str, Any] | str]:
    text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
    payloads: list[dict[str, Any] | str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data_text = line[5:].strip()
        if not data_text:
            continue
        if data_text == "[DONE]":
            payloads.append("[DONE]")
            continue
        try:
            parsed = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads


def extract_upstream_turn_error_from_chunk(chunk: bytes | str) -> UpstreamTurnError | None:
    for payload in parse_sse_chunk_payloads(chunk):
        if not isinstance(payload, dict) or not payload.get("turn_complete"):
            continue
        error_code = str(payload.get("error_code") or "").strip()
        error_message = str(payload.get("error_message") or "").strip()
        if not error_code and not error_message:
            continue
        return UpstreamTurnError(
            error_code=error_code,
            error_message=error_message,
            payload=payload,
        )
    return None


def native_sse_chunk_has_meaningful_output(chunk: bytes | str) -> bool:
    if extract_upstream_turn_error_from_chunk(chunk) is not None:
        return False
    return any(
        payload != "[DONE]"
        for payload in parse_sse_chunk_payloads(chunk)
    )


def anthropic_stream_chunk_has_meaningful_output(chunk: bytes | str) -> bool:
    text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
    event_name = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("event:"):
            event_name = line[6:].strip()
            break

    payloads = [
        payload
        for payload in parse_sse_chunk_payloads(chunk)
        if isinstance(payload, dict)
    ]
    if not payloads:
        return False
    payload = payloads[0]
    if event_name == "content_block_start":
        block = payload.get("content_block")
        return isinstance(block, dict) and str(block.get("type") or "") == "tool_use"
    if event_name == "content_block_delta":
        delta = payload.get("delta")
        if not isinstance(delta, dict):
            return False
        return bool(delta.get("text")) or str(delta.get("type") or "") == "input_json_delta"
    return False


def openai_chat_chunk_has_meaningful_output(chunk: bytes | str) -> bool:
    for payload in parse_sse_chunk_payloads(chunk):
        if not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if str(delta.get("content") or ""):
                return True
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                return True
    return False


def openai_responses_chunk_has_meaningful_output(chunk: bytes | str) -> bool:
    for payload in parse_sse_chunk_payloads(chunk):
        if not isinstance(payload, dict):
            continue
        event_type = str(payload.get("type") or "")
        if event_type == "response.output_text.delta" and str(payload.get("delta") or ""):
            return True
        if event_type == "response.output_item.added":
            item = payload.get("item")
            if isinstance(item, dict) and str(item.get("type") or "") == "tool_call":
                return True
    return False


def gemini_stream_chunk_has_meaningful_output(chunk: bytes | str) -> bool:
    for payload in parse_sse_chunk_payloads(chunk):
        if not isinstance(payload, dict):
            continue
        return not bool(summarize_gemini_response(payload)["empty_response"])
    return False


def prefetch_stream_until_meaningful(
    stream_bytes: Iterator[bytes],
    *,
    chunk_has_meaningful_output: Callable[[bytes | str], bool],
) -> tuple[list[bytes], Iterator[bytes], bool]:
    prefetched: list[bytes] = []
    for chunk in stream_bytes:
        prefetched.append(chunk)
        if chunk_has_meaningful_output(chunk):
            return prefetched, stream_bytes, True
    return prefetched, stream_bytes, False
