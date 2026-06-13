from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import requests
from fastapi.responses import Response

from ..api_logs import ApiLogStore
from ..app_settings import PanelSettings
from ..models import Account
from ..upstream_support import (
    is_stream_summary_empty,
    prefetch_stream_until_meaningful,
    record_proxy_log,
)
from ..usage_stats import UsageStatsStore


@dataclass(slots=True)
class ProxyEndpointConfig:
    event: str
    model: str
    default_stop_reason: str
    stream_complete_message: str
    error_response_builder: Callable[..., Response]
    extra_fields_extractor: Callable[[dict[str, Any]], dict[str, Any]]
    include_remaining_header: bool = True
    max_tokens: int | None = None
    disable_on_empty_response: bool = True
    use_stream_summary_empty_check: bool = True
    provider: str = ""
    cache_token_fields: list[str] = field(default_factory=list)


def make_record_final_log(
    *,
    config: ProxyEndpointConfig,
    api_log_store: ApiLogStore,
    panel_settings: PanelSettings,
    started_at: float,
    messages_count: int,
) -> Callable[..., None]:
    def _record_final_log(
        selected_account: Account,
        selected_quota: dict[str, Any],
        selected_request_id: str,
        *,
        stream: bool,
        success: bool,
        stop_reason: str,
        message: str,
        status_code: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        empty_response: bool = False,
        level: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        record_proxy_log(
            api_log_store,
            event=config.event,
            model=config.model,
            stream=stream,
            strategy=panel_settings.api_account_strategy,
            request_id=selected_request_id,
            success=success,
            stop_reason=stop_reason,
            message=message,
            status_code=status_code,
            account=selected_account,
            quota=selected_quota,
            empty_response=empty_response,
            messages_count=messages_count,
            max_tokens=config.max_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            level=level,
            extra_fields=extra_fields,
        )

    return _record_final_log


def make_build_upstream_error_response(
    *,
    config: ProxyEndpointConfig,
    record_final_log: Callable[..., None],
) -> Callable[..., Callable[[int, str, str], Response]]:
    error_builder = config.error_response_builder

    def _build_upstream_error_response(
        selected_account: Account,
        selected_quota: dict[str, Any],
        selected_request_id: str,
        *,
        stream: bool,
    ) -> Callable[[int, str, str], Response]:
        def _handle(status_code: int, message: str, stop_reason: str) -> Response:
            record_final_log(
                selected_account,
                selected_quota,
                selected_request_id,
                stream=stream,
                success=False,
                stop_reason=stop_reason,
                message=message,
                status_code=status_code,
            )
            return error_builder(status_code, message, stop_reason)

        return _handle

    return _build_upstream_error_response


def make_stream_headers(
    *,
    panel_settings: PanelSettings,
    include_remaining: bool = True,
) -> Callable[[Account, dict[str, Any]], dict[str, str]]:
    def _stream_headers(
        selected_account: Account,
        selected_quota: dict[str, Any],
    ) -> dict[str, str]:
        headers = {
            "x-accio-account-id": selected_account.id,
            "x-accio-account-strategy": panel_settings.api_account_strategy,
        }
        if include_remaining and selected_quota.get("success"):
            headers["x-accio-account-remaining"] = str(
                selected_quota["remaining_value"]
            )
        return headers

    return _stream_headers


def make_build_stream_attempt(
    *,
    config: ProxyEndpointConfig,
    panel_settings: PanelSettings,
    store: Any,
    usage_stats_store: UsageStatsStore,
    api_log_store: ApiLogStore,
    started_at: float,
    messages_count: int,
    record_attempt: Callable[..., None],
    disable_account_model_on_empty_response: Callable[..., None],
    clear_account_sentinel_rate_limit: Callable[..., None],
    empty_response_log_message: Callable[..., str],
    iter_sse_bytes: Callable[..., Iterator[bytes]],
    chunk_has_meaningful_output: Callable[[bytes | str], bool],
    iter_sse_extra_kwargs: Callable[..., dict[str, Any]] | None = None,
) -> Callable[..., tuple[Iterator[bytes], bool]]:
    def _build_stream_attempt(
        selected_account: Account,
        selected_quota: dict[str, Any],
        selected_response: requests.Response,
        selected_request_id: str,
        selected_attempt: int,
        selected_attempt_started_at: float,
    ) -> tuple[Iterator[bytes], bool]:
        def on_stream_complete(summary: dict[str, Any]) -> None:
            usage = summary.get("usage") if isinstance(summary.get("usage"), dict) else {}
            if not isinstance(usage, dict):
                usage = {}
            empty_response = (
                is_stream_summary_empty(summary)
                if config.use_stream_summary_empty_check
                else bool(summary.get("empty_response"))
            )
            if empty_response and config.disable_on_empty_response:
                kwargs: dict[str, Any] = {}
                if config.provider:
                    kwargs["provider"] = config.provider
                disable_account_model_on_empty_response(
                    store,
                    selected_account,
                    config.model,
                    **kwargs,
                )
            if not empty_response:
                clear_account_sentinel_rate_limit(store, selected_account)
            stop_reason = (
                "empty_response"
                if empty_response
                else str(summary.get("stop_reason") or config.default_stop_reason)
            )
            record_attempt(
                selected_account,
                selected_quota,
                selected_request_id,
                attempt=selected_attempt,
                stream=True,
                success=not empty_response,
                stop_reason=stop_reason,
                message=(
                    empty_response_log_message(
                        config.model,
                        disable_model=config.disable_on_empty_response,
                    )
                    if empty_response
                    else config.stream_complete_message
                ),
                status_code=200,
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                empty_response=empty_response,
                duration_ms=int(
                    (time.perf_counter() - selected_attempt_started_at) * 1000
                ),
                level="warn" if empty_response else None,
                extra_fields=config.extra_fields_extractor(summary),
            )
            cache_kwargs: dict[str, int] = {}
            for cache_field in config.cache_token_fields:
                cache_kwargs[cache_field] = int(usage.get(cache_field) or 0)
            usage_stats_store.record_message(
                account_id=selected_account.id,
                model=config.model,
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                success=True,
                stop_reason=str(summary.get("stop_reason") or config.default_stop_reason),
                **cache_kwargs,
            )
            log_entry: dict[str, Any] = {
                "level": "warn" if empty_response else "info",
                "event": config.event,
                "success": True,
                "emptyResponse": empty_response,
                "accountId": selected_account.id,
                "accountName": selected_account.name,
                "fillPriority": selected_account.fill_priority,
                "model": config.model,
                "stream": True,
                "strategy": panel_settings.api_account_strategy,
                "requestId": selected_request_id,
                "message": (
                    empty_response_log_message(
                        config.model,
                        disable_model=True,
                    )
                    if empty_response
                    else config.stream_complete_message
                ),
                "statusCode": 200,
                "stopReason": str(summary.get("stop_reason") or config.default_stop_reason),
                "inputTokens": int(usage.get("input_tokens") or 0),
                "outputTokens": int(usage.get("output_tokens") or 0),
                "remainingQuota": selected_quota.get("remaining_value"),
                "usedQuota": selected_quota.get("used_value"),
                "messagesCount": messages_count,
                "durationMs": int((time.perf_counter() - started_at) * 1000),
            }
            if config.max_tokens is not None:
                log_entry["maxTokens"] = config.max_tokens
            for cache_field in config.cache_token_fields:
                camel = _snake_to_camel(cache_field)
                log_entry[camel] = int(usage.get(cache_field) or 0)
            extra = config.extra_fields_extractor(summary)
            for key, value in extra.items():
                log_entry[key] = value
            api_log_store.record(log_entry)

        extra_kwargs = {}
        if iter_sse_extra_kwargs is not None:
            extra_kwargs = iter_sse_extra_kwargs(
                selected_account, selected_quota, selected_request_id,
            )
        stream_bytes = iter_sse_bytes(
            selected_response,
            config.model,
            on_complete=on_stream_complete,
            **extra_kwargs,
        )
        prefetched_chunks, remaining_chunks, has_meaningful_output = (
            prefetch_stream_until_meaningful(
                stream_bytes,
                chunk_has_meaningful_output=chunk_has_meaningful_output,
            )
        )
        return (
            itertools.chain(prefetched_chunks, remaining_chunks),
            has_meaningful_output,
        )

    return _build_stream_attempt


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])
