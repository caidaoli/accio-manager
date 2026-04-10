from __future__ import annotations

import asyncio
import itertools
import json
import time
from typing import Any, Callable, Iterator

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..api_logs import ApiLogStore
from ..anthropic_proxy import UpstreamTurnError
from ..gemini_proxy import (
    build_generate_content_request,
    extract_gemini_finish_reason,
    extract_gemini_usage,
    gemini_error_payload,
    iter_gemini_generate_content_sse_bytes,
    normalize_gemini_model_name,
    summarize_gemini_response,
)
from ..models import Account
from ..upstream_support import (
    extract_upstream_turn_error_from_chunk as _extract_upstream_turn_error_from_chunk,
    gemini_stream_chunk_has_meaningful_output as _gemini_stream_chunk_has_meaningful_output,
    is_stream_summary_empty as _is_stream_summary_empty,
    make_upstream_attempt_logger as _make_upstream_attempt_logger,
    native_sse_chunk_has_meaningful_output as _native_sse_chunk_has_meaningful_output,
    prefetch_stream_until_meaningful as _prefetch_stream_until_meaningful,
    record_proxy_log,
    request_upstream_or_error,
    should_retry_upstream_turn_error as _should_retry_upstream_turn_error,
    summarize_non_stream_payload as _summarize_non_stream_payload,
    upstream_turn_error_message as _upstream_turn_error_message,
)
from ..usage_stats import UsageStatsStore
from .context import ProxyRouteContext


def install_gemini_routes(context: ProxyRouteContext) -> None:
    application = context.application
    store = context.store
    client = context.client
    panel_settings_store = context.panel_settings_store
    usage_stats_store: UsageStatsStore = context.usage_stats_store
    api_log_store: ApiLogStore = context.api_log_store
    ProxySelectionError = context.ProxySelectionError

    _authorize_proxy_request = context.authorize_proxy_request
    _is_allowed_dynamic_model = context.is_allowed_dynamic_model
    _select_proxy_account = context.select_proxy_account
    _empty_response_log_message = context.empty_response_log_message
    _should_disable_model_on_empty_response = context.should_disable_model_on_empty_response
    _disable_account_model_on_empty_response = context.disable_account_model_on_empty_response
    _extract_proxy_api_key = context.extract_proxy_api_key
    _iter_upstream_sse_bytes = context.iter_upstream_sse_bytes
    _gemini_error_response = context.gemini_error_response
    _decode_gemini_generate_content_response = context.decode_gemini_generate_content_response
    _native_error_response = context.native_error_response

    async def _handle_gemini_generate_content(
        request: Request,
        model_name: str,
        *,
        force_stream: bool = False,
    ) -> Response:
        panel_settings = panel_settings_store.load()
        usage_stats_store: UsageStatsStore = application.state.usage_stats_store
        api_log_store: ApiLogStore = application.state.api_log_store
        unauthorized = _authorize_proxy_request(request, panel_settings)
        if unauthorized:
            return JSONResponse(
                gemini_error_payload(401, "无效的 API Key", error_status="UNAUTHENTICATED"),
                status_code=401,
            )

        normalized_model_name = normalize_gemini_model_name(model_name)
        allowed, available = _is_allowed_dynamic_model(
            application,
            panel_settings,
            normalized_model_name,
            provider="gemini",
        )
        if not allowed:
            return _gemini_error_response(
                400,
                f"不支持模型 {normalized_model_name}。当前可用 Gemini 模型: {', '.join(available)}",
            )

        started_at = time.perf_counter()
        requested_stream = force_stream or str(request.query_params.get("alt") or "").strip().lower() == "sse"
        raw_body = await request.body()
        if not raw_body.strip():
            return _gemini_error_response(400, "请求体不能为空。")

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _gemini_error_response(400, "请求体必须是合法的 JSON。")
        if not isinstance(payload, dict):
            return _gemini_error_response(400, "请求体必须是 JSON 对象。")

        contents_value = payload.get("contents")
        messages_count = len(contents_value) if isinstance(contents_value, list) else 0

        try:
            account, quota = await asyncio.to_thread(
                _select_proxy_account,
                application,
                panel_settings,
                normalized_model_name,
                provider="gemini",
            )
        except ProxySelectionError as exc:
            api_log_store.record(
                {
                    "level": "error",
                    "event": "gemini_generate_content",
                    "success": False,
                    "emptyResponse": False,
                    "accountId": "",
                    "accountName": "-",
                    "fillPriority": None,
                    "model": normalized_model_name,
                    "stream": requested_stream,
                    "strategy": panel_settings.api_account_strategy,
                    "requestId": "",
                    "message": exc.message,
                    "statusCode": exc.status_code,
                    "stopReason": "proxy_selection_failed",
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "remainingQuota": None,
                    "usedQuota": None,
                    "messagesCount": messages_count,
                    "durationMs": int((time.perf_counter() - started_at) * 1000),
                }
            )
            return _gemini_error_response(
                exc.status_code,
                exc.message,
                error_status="UNAVAILABLE",
            )

        accio_body = build_generate_content_request(
            payload,
            model=normalized_model_name,
            token=account.access_token,
        )
        request_id = str(accio_body.get("request_id") or "")
        root_request_id = request_id

        _record_attempt = _make_upstream_attempt_logger(
            api_log_store,
            event="gemini_generate_content",
            model=normalized_model_name,
            strategy=panel_settings.api_account_strategy,
            root_request_id=root_request_id,
            messages_count=messages_count,
        )

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
                event="gemini_generate_content",
                model=normalized_model_name,
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
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                level=level,
                extra_fields=extra_fields,
            )

        def _build_upstream_error_response(
            selected_account: Account,
            selected_quota: dict[str, Any],
            selected_request_id: str,
            *,
            stream: bool,
        ) -> Callable[[int, str, str], Response]:
            def _handle(status_code: int, message: str, stop_reason: str) -> Response:
                _record_final_log(
                    selected_account,
                    selected_quota,
                    selected_request_id,
                    stream=stream,
                    success=False,
                    stop_reason=stop_reason,
                    message=message,
                    status_code=status_code,
                )
                return _gemini_error_response(
                    status_code,
                    message,
                    error_status="UNAVAILABLE",
                )

            return _handle

        attempt_started_at = time.perf_counter()
        upstream_response = await request_upstream_or_error(
            lambda: asyncio.to_thread(
                client.generate_content,
                account,
                accio_body,
                proxy_url=panel_settings.upstream_proxy_url,
            ),
            account=account,
            quota=quota,
            request_id=request_id,
            attempt=1,
            stream=requested_stream,
            started_at=attempt_started_at,
            record_attempt=_record_attempt,
            build_error_response=_build_upstream_error_response(
                account,
                quota,
                request_id,
                stream=requested_stream,
            ),
            usage_failure_recorder=lambda stop_reason: usage_stats_store.record_message(
                account_id=account.id,
                model=normalized_model_name,
                input_tokens=0,
                output_tokens=0,
                success=False,
                stop_reason=stop_reason,
            ),
        )
        if isinstance(upstream_response, Response):
            return upstream_response

        if requested_stream:
            def _stream_headers(
                selected_account: Account,
                selected_quota: dict[str, Any],
            ) -> dict[str, str]:
                return {
                    "x-accio-account-id": selected_account.id,
                    "x-accio-account-strategy": panel_settings.api_account_strategy,
                }

            def _build_stream_attempt(
                selected_account: Account,
                selected_quota: dict[str, Any],
                selected_response: requests.Response,
                selected_request_id: str,
                selected_attempt: int,
                selected_attempt_started_at: float,
            ) -> tuple[Iterator[bytes], bool]:
                def on_gemini_stream_complete(summary: dict[str, Any]) -> None:
                    usage = summary.get("usage") if isinstance(summary.get("usage"), dict) else {}
                    empty_response = bool(summary.get("empty_response"))
                    if empty_response:
                        _disable_account_model_on_empty_response(
                            store,
                            selected_account,
                            normalized_model_name,
                            provider="gemini",
                        )
                    _record_attempt(
                        selected_account,
                        selected_quota,
                        selected_request_id,
                        attempt=selected_attempt,
                        stream=True,
                        success=not empty_response,
                        stop_reason=(
                            "empty_response"
                            if empty_response
                            else str(summary.get("stop_reason") or "STOP")
                        ),
                        message=(
                            _empty_response_log_message(
                                normalized_model_name,
                                disable_model=True,
                            )
                            if empty_response
                            else "Gemini 上游流式请求完成"
                        ),
                        status_code=200,
                        input_tokens=int(usage.get("input_tokens") or 0),
                        output_tokens=int(usage.get("output_tokens") or 0),
                        empty_response=empty_response,
                        duration_ms=int(
                            (time.perf_counter() - selected_attempt_started_at) * 1000
                        ),
                        extra_fields={
                            "imageBlocks": int(summary.get("image_blocks") or 0),
                            "textChars": int(summary.get("text_chars") or 0),
                            "toolUseBlocks": int(summary.get("tool_use_blocks") or 0),
                        },
                    )
                    usage_stats_store.record_message(
                        account_id=selected_account.id,
                        model=normalized_model_name,
                        input_tokens=int(usage.get("input_tokens") or 0),
                        output_tokens=int(usage.get("output_tokens") or 0),
                        success=True,
                        stop_reason=str(summary.get("stop_reason") or "STOP"),
                    )
                    api_log_store.record(
                        {
                            "level": "warn" if empty_response else "info",
                            "event": "gemini_generate_content",
                            "success": True,
                            "emptyResponse": empty_response,
                            "accountId": selected_account.id,
                            "accountName": selected_account.name,
                            "fillPriority": selected_account.fill_priority,
                            "model": normalized_model_name,
                            "stream": True,
                            "strategy": panel_settings.api_account_strategy,
                            "requestId": selected_request_id,
                            "message": (
                                _empty_response_log_message(
                                    normalized_model_name,
                                    disable_model=True,
                                )
                                if empty_response
                                else "Gemini 流式兼容调用完成"
                            ),
                            "statusCode": 200,
                            "stopReason": str(summary.get("stop_reason") or "STOP"),
                            "inputTokens": int(usage.get("input_tokens") or 0),
                            "outputTokens": int(usage.get("output_tokens") or 0),
                            "remainingQuota": selected_quota.get("remaining_value"),
                            "usedQuota": selected_quota.get("used_value"),
                            "messagesCount": messages_count,
                            "imageBlocks": int(summary.get("image_blocks") or 0),
                            "textChars": int(summary.get("text_chars") or 0),
                            "toolUseBlocks": int(summary.get("tool_use_blocks") or 0),
                            "durationMs": int((time.perf_counter() - started_at) * 1000),
                        }
                    )

                stream_bytes = iter_gemini_generate_content_sse_bytes(
                    selected_response,
                    normalized_model_name,
                    on_complete=on_gemini_stream_complete,
                )
                prefetched_chunks, remaining_chunks, has_meaningful_output = (
                    _prefetch_stream_until_meaningful(
                        stream_bytes,
                        chunk_has_meaningful_output=_gemini_stream_chunk_has_meaningful_output,
                    )
                )
                return (
                    itertools.chain(prefetched_chunks, remaining_chunks),
                    has_meaningful_output,
                )

            stream_account = account
            stream_quota = quota
            try:
                stream_bytes, has_meaningful_output = _build_stream_attempt(
                    stream_account,
                    stream_quota,
                    upstream_response,
                    request_id,
                    1,
                    attempt_started_at,
                )
            except UpstreamTurnError as exc:
                _record_attempt(
                    stream_account,
                    stream_quota,
                    request_id,
                    attempt=1,
                    stream=True,
                    success=False,
                    stop_reason="upstream_turn_error",
                    message=_upstream_turn_error_message(exc),
                    status_code=502,
                    duration_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                    extra_fields={"errorCode": exc.error_code or None},
                )
                if not _should_retry_upstream_turn_error(exc):
                    return _anthropic_error_response(
                        502,
                        _upstream_turn_error_message(exc),
                    )
                has_meaningful_output = False
            if not has_meaningful_output:
                try:
                    retry_account, retry_quota = await asyncio.to_thread(
                        _select_proxy_account,
                        application,
                        panel_settings,
                        normalized_model_name,
                        provider="gemini",
                        exclude_account_ids={stream_account.id},
                    )
                except ProxySelectionError as exc:
                    return _gemini_error_response(
                        exc.status_code,
                        exc.message,
                        error_status="UNAVAILABLE",
                    )

                retry_body = build_generate_content_request(
                    payload,
                    model=normalized_model_name,
                    token=retry_account.access_token,
                )
                retry_request_id = str(retry_body.get("request_id") or "")
                retry_started_at = time.perf_counter()
                retry_response = await request_upstream_or_error(
                    lambda: asyncio.to_thread(
                        client.generate_content,
                        retry_account,
                        retry_body,
                        proxy_url=panel_settings.upstream_proxy_url,
                    ),
                    account=retry_account,
                    quota=retry_quota,
                    request_id=retry_request_id,
                    attempt=2,
                    stream=True,
                    started_at=retry_started_at,
                    record_attempt=_record_attempt,
                    build_error_response=_build_upstream_error_response(
                        retry_account,
                        retry_quota,
                        retry_request_id,
                        stream=True,
                    ),
                    retry_reason="upstream_turn_error_or_empty_response",
                )
                if isinstance(retry_response, Response):
                    return retry_response

                stream_account = retry_account
                stream_quota = retry_quota
                try:
                    stream_bytes, _ = _build_stream_attempt(
                        stream_account,
                        stream_quota,
                        retry_response,
                        retry_request_id,
                        2,
                        retry_started_at,
                    )
                except UpstreamTurnError as exc:
                    _record_attempt(
                        stream_account,
                        stream_quota,
                        retry_request_id,
                        attempt=2,
                        stream=True,
                        success=False,
                        stop_reason="upstream_turn_error",
                        message=_upstream_turn_error_message(exc),
                        status_code=502,
                        duration_ms=int((time.perf_counter() - retry_started_at) * 1000),
                        extra_fields={
                            "errorCode": exc.error_code or None,
                            "retryReason": "upstream_turn_error_or_empty_response",
                        },
                    )
                    return _anthropic_error_response(
                        502,
                        _upstream_turn_error_message(exc),
                    )

            return StreamingResponse(
                stream_bytes,
                media_type="text/event-stream",
                headers=_stream_headers(stream_account, stream_quota),
            )

        try:
            response_payload = await asyncio.to_thread(
                _decode_gemini_generate_content_response,
                upstream_response,
                normalized_model_name,
            )
        except ValueError as exc:
            usage_stats_store.record_message(
                account_id=account.id,
                model=normalized_model_name,
                input_tokens=0,
                output_tokens=0,
                success=False,
                stop_reason="decode_error",
            )
            _record_attempt(
                account,
                quota,
                request_id,
                attempt=1,
                stream=False,
                success=False,
                stop_reason="decode_error",
                message=str(exc),
                status_code=502,
                duration_ms=int((time.perf_counter() - attempt_started_at) * 1000),
            )
            api_log_store.record(
                {
                    "level": "error",
                    "event": "gemini_generate_content",
                    "success": False,
                    "emptyResponse": False,
                    "accountId": account.id,
                    "accountName": account.name,
                    "fillPriority": account.fill_priority,
                    "model": normalized_model_name,
                    "stream": False,
                    "strategy": panel_settings.api_account_strategy,
                    "requestId": request_id,
                    "message": str(exc),
                    "statusCode": 502,
                    "stopReason": "decode_error",
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "remainingQuota": quota.get("remaining_value"),
                    "usedQuota": quota.get("used_value"),
                    "messagesCount": messages_count,
                    "durationMs": int((time.perf_counter() - started_at) * 1000),
                }
            )
            return _gemini_error_response(502, str(exc), error_status="UNAVAILABLE")

        usage = extract_gemini_usage(response_payload)
        output_summary = summarize_gemini_response(response_payload)
        current_attempt = 1
        current_attempt_started_at = attempt_started_at
        if output_summary["empty_response"]:
            _disable_account_model_on_empty_response(
                store,
                account,
                normalized_model_name,
                provider="gemini",
            )
            finish_reason = extract_gemini_finish_reason(response_payload)
            _record_attempt(
                account,
                quota,
                request_id,
                attempt=1,
                stream=False,
                success=False,
                stop_reason="empty_response",
                message=_empty_response_log_message(
                    normalized_model_name,
                    disable_model=True,
                ),
                status_code=200,
                input_tokens=int(usage["input_tokens"]),
                output_tokens=int(usage["output_tokens"]),
                empty_response=True,
                duration_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                level="warn",
                extra_fields={
                    "textChars": int(output_summary["text_chars"]),
                    "toolUseBlocks": int(output_summary["tool_use_blocks"]),
                    "imageBlocks": int(output_summary["image_blocks"]),
                },
            )
            usage_stats_store.record_message(
                account_id=account.id,
                model=normalized_model_name,
                input_tokens=int(usage["input_tokens"]),
                output_tokens=int(usage["output_tokens"]),
                success=True,
                stop_reason=finish_reason,
            )
            api_log_store.record(
                {
                    "level": "warn",
                    "event": "gemini_generate_content",
                    "success": True,
                    "emptyResponse": True,
                    "accountId": account.id,
                    "accountName": account.name,
                    "fillPriority": account.fill_priority,
                    "model": normalized_model_name,
                    "stream": False,
                    "strategy": panel_settings.api_account_strategy,
                    "requestId": request_id,
                    "message": _empty_response_log_message(
                        normalized_model_name,
                        disable_model=True,
                    ),
                    "statusCode": 200,
                    "stopReason": finish_reason,
                    "inputTokens": int(usage["input_tokens"]),
                    "outputTokens": int(usage["output_tokens"]),
                    "remainingQuota": quota.get("remaining_value"),
                    "usedQuota": quota.get("used_value"),
                    "messagesCount": messages_count,
                    "textChars": int(output_summary["text_chars"]),
                    "toolUseBlocks": int(output_summary["tool_use_blocks"]),
                    "imageBlocks": int(output_summary["image_blocks"]),
                    "durationMs": int((time.perf_counter() - started_at) * 1000),
                }
            )
            try:
                retry_account, retry_quota = await asyncio.to_thread(
                    _select_proxy_account,
                    application,
                    panel_settings,
                    normalized_model_name,
                    provider="gemini",
                    exclude_account_ids={account.id},
                )
            except ProxySelectionError as exc:
                return _gemini_error_response(
                    exc.status_code,
                    exc.message,
                    error_status="UNAVAILABLE",
                )

            retry_body = build_generate_content_request(
                payload,
                model=normalized_model_name,
                token=retry_account.access_token,
            )
            retry_request_id = str(retry_body.get("request_id") or "")
            retry_started_at = time.perf_counter()
            retry_response = await request_upstream_or_error(
                lambda: asyncio.to_thread(
                    client.generate_content,
                    retry_account,
                    retry_body,
                    proxy_url=panel_settings.upstream_proxy_url,
                ),
                account=retry_account,
                quota=retry_quota,
                request_id=retry_request_id,
                attempt=2,
                stream=False,
                started_at=retry_started_at,
                record_attempt=_record_attempt,
                build_error_response=_build_upstream_error_response(
                    retry_account,
                    retry_quota,
                    retry_request_id,
                    stream=False,
                ),
                retry_reason="empty_response",
            )
            if isinstance(retry_response, Response):
                return retry_response

            account = retry_account
            quota = retry_quota
            request_id = retry_request_id
            current_attempt = 2
            current_attempt_started_at = retry_started_at
            try:
                response_payload = await asyncio.to_thread(
                    _decode_gemini_generate_content_response,
                    retry_response,
                    normalized_model_name,
                )
            except ValueError as exc:
                _record_attempt(
                    account,
                    quota,
                    request_id,
                    attempt=2,
                    stream=False,
                    success=False,
                    stop_reason="decode_error",
                    message=str(exc),
                    status_code=502,
                    duration_ms=int((time.perf_counter() - retry_started_at) * 1000),
                    extra_fields={"retryReason": "empty_response"},
                )
                return _gemini_error_response(
                    502,
                    str(exc),
                    error_status="UNAVAILABLE",
                )
            usage = extract_gemini_usage(response_payload)
            output_summary = summarize_gemini_response(response_payload)
            if output_summary["empty_response"]:
                _disable_account_model_on_empty_response(
                    store,
                    account,
                    normalized_model_name,
                    provider="gemini",
                )
        finish_reason = extract_gemini_finish_reason(response_payload)
        _record_attempt(
            account,
            quota,
            request_id,
            attempt=current_attempt,
            stream=False,
            success=not bool(output_summary["empty_response"]),
            stop_reason=(
                "empty_response"
                if output_summary["empty_response"]
                else finish_reason
            ),
            message=(
                _empty_response_log_message(
                    normalized_model_name,
                    disable_model=True,
                )
                if output_summary["empty_response"]
                else "Gemini 上游请求完成"
            ),
            status_code=200,
            input_tokens=int(usage["input_tokens"]),
            output_tokens=int(usage["output_tokens"]),
            empty_response=bool(output_summary["empty_response"]),
            duration_ms=int((time.perf_counter() - current_attempt_started_at) * 1000),
            level="warn" if output_summary["empty_response"] else None,
            extra_fields={
                "textChars": int(output_summary["text_chars"]),
                "toolUseBlocks": int(output_summary["tool_use_blocks"]),
                "imageBlocks": int(output_summary["image_blocks"]),
            },
        )
        usage_stats_store.record_message(
            account_id=account.id,
            model=normalized_model_name,
            input_tokens=int(usage["input_tokens"]),
            output_tokens=int(usage["output_tokens"]),
            success=True,
            stop_reason=finish_reason,
        )
        api_log_store.record(
            {
                "level": "warn" if output_summary["empty_response"] else "info",
                "event": "gemini_generate_content",
                "success": True,
                "emptyResponse": bool(output_summary["empty_response"]),
                "accountId": account.id,
                "accountName": account.name,
                "fillPriority": account.fill_priority,
                "model": normalized_model_name,
                "stream": False,
                "strategy": panel_settings.api_account_strategy,
                "requestId": request_id,
                "message": (
                    _empty_response_log_message(
                        normalized_model_name,
                        disable_model=True,
                    )
                    if output_summary["empty_response"]
                    else "Gemini 兼容调用完成"
                ),
                "statusCode": 200,
                "stopReason": finish_reason,
                "inputTokens": int(usage["input_tokens"]),
                "outputTokens": int(usage["output_tokens"]),
                "remainingQuota": quota.get("remaining_value"),
                "usedQuota": quota.get("used_value"),
                "messagesCount": messages_count,
                "textChars": int(output_summary["text_chars"]),
                "toolUseBlocks": int(output_summary["tool_use_blocks"]),
                "imageBlocks": int(output_summary["image_blocks"]),
                "durationMs": int((time.perf_counter() - started_at) * 1000),
            }
        )
        response_headers = {
            "x-accio-account-id": account.id,
            "x-accio-account-strategy": panel_settings.api_account_strategy,
        }
        if quota.get("success"):
            response_headers["x-accio-account-remaining"] = str(
                quota["remaining_value"]
            )
        return JSONResponse(response_payload, headers=response_headers)

    @application.post("/v1beta/models/{model_name}:streamGenerateContent")
    async def gemini_stream_generate_content(
        request: Request,
        model_name: str,
    ) -> Response:
        return await _handle_gemini_generate_content(
            request,
            model_name,
            force_stream=True,
        )

    @application.post("/v1beta/models/{model_name}:generateContent")
    async def gemini_generate_content(request: Request, model_name: str) -> Response:
        return await _handle_gemini_generate_content(request, model_name)

    @application.post("/api/adk/llm/generateContent")
    async def native_generate_content(request: Request) -> Response:
        panel_settings = panel_settings_store.load()
        api_log_store: ApiLogStore = application.state.api_log_store
        unauthorized = _authorize_proxy_request(request, panel_settings)
        if unauthorized:
            return unauthorized

        raw_body = await request.body()
        if not raw_body.strip():
            return _native_error_response(400, "请求体不能为空。")

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _native_error_response(400, "请求体必须是合法的 JSON。")

        if not isinstance(payload, dict):
            return _native_error_response(400, "请求体必须是 JSON 对象。")

        model = str(payload.get("model") or "").strip()
        if not model:
            return _native_error_response(400, "请求体缺少 model。")

        allowed, available = _is_allowed_dynamic_model(
            application,
            panel_settings,
            model,
        )
        if not allowed:
            return _native_error_response(
                400,
                f"不支持模型 {model}。当前可用模型: {', '.join(available)}",
            )

        try:
            account, quota = await asyncio.to_thread(
                _select_proxy_account,
                application,
                panel_settings,
                model,
            )
        except ProxySelectionError as exc:
            return _native_error_response(exc.status_code, exc.message)

        accio_body = build_generate_content_request(
            payload,
            token=account.access_token,
        )
        request_id = str(accio_body.get("request_id") or "")
        root_request_id = request_id
        _record_attempt = _make_upstream_attempt_logger(
            api_log_store,
            event="native_generate_content",
            model=model,
            strategy=panel_settings.api_account_strategy,
            root_request_id=root_request_id,
        )

        def _build_upstream_error_response() -> Callable[[int, str, str], Response]:
            def _handle(status_code: int, message: str, _stop_reason: str) -> Response:
                return _native_error_response(status_code, message)

            return _handle

        attempt_started_at = time.perf_counter()
        upstream_response = await request_upstream_or_error(
            lambda: asyncio.to_thread(
                client.generate_content,
                account,
                accio_body,
                proxy_url=panel_settings.upstream_proxy_url,
            ),
            account=account,
            quota=quota,
            request_id=request_id,
            attempt=1,
            stream=True,
            started_at=attempt_started_at,
            record_attempt=_record_attempt,
            build_error_response=_build_upstream_error_response(),
        )
        if isinstance(upstream_response, Response):
            return upstream_response

        stream_account = account
        stream_quota = quota
        stream_response = upstream_response
        prefetched_chunks, remaining_chunks, _ = _prefetch_stream_until_meaningful(
            _iter_upstream_sse_bytes(upstream_response),
            chunk_has_meaningful_output=_native_sse_chunk_has_meaningful_output,
        )
        turn_error = next(
            (
                error
                for error in (
                    _extract_upstream_turn_error_from_chunk(chunk)
                    for chunk in prefetched_chunks
                )
                if error is not None
            ),
            None,
        )
        if turn_error is not None:
            _record_attempt(
                account,
                quota,
                request_id,
                attempt=1,
                success=False,
                stream=True,
                stop_reason="upstream_turn_error",
                message=_upstream_turn_error_message(turn_error),
                status_code=502,
                duration_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                extra_fields={"errorCode": turn_error.error_code or None},
            )
            if not _should_retry_upstream_turn_error(turn_error):
                return _native_error_response(
                    502,
                    _upstream_turn_error_message(turn_error),
                )
            try:
                retry_account, retry_quota = await asyncio.to_thread(
                    _select_proxy_account,
                    application,
                    panel_settings,
                    model,
                    exclude_account_ids={account.id},
                )
            except ProxySelectionError as exc:
                return _native_error_response(exc.status_code, exc.message)

            retry_body = build_generate_content_request(
                payload,
                token=retry_account.access_token,
            )
            retry_request_id = str(retry_body.get("request_id") or "")
            retry_started_at = time.perf_counter()
            retry_response = await request_upstream_or_error(
                lambda: asyncio.to_thread(
                    client.generate_content,
                    retry_account,
                    retry_body,
                    proxy_url=panel_settings.upstream_proxy_url,
                ),
                account=retry_account,
                quota=retry_quota,
                request_id=retry_request_id,
                attempt=2,
                stream=True,
                started_at=retry_started_at,
                record_attempt=_record_attempt,
                build_error_response=_build_upstream_error_response(),
                retry_reason="upstream_turn_error",
            )
            if isinstance(retry_response, Response):
                return retry_response

            stream_account = retry_account
            stream_quota = retry_quota
            stream_response = retry_response
            prefetched_chunks, remaining_chunks, _ = _prefetch_stream_until_meaningful(
                _iter_upstream_sse_bytes(retry_response),
                chunk_has_meaningful_output=_native_sse_chunk_has_meaningful_output,
            )
            retry_turn_error = next(
                (
                    error
                    for error in (
                        _extract_upstream_turn_error_from_chunk(chunk)
                        for chunk in prefetched_chunks
                    )
                    if error is not None
                ),
                None,
            )
            if retry_turn_error is not None:
                _record_attempt(
                    stream_account,
                    stream_quota,
                    retry_request_id,
                    attempt=2,
                    success=False,
                    stream=True,
                    stop_reason="upstream_turn_error",
                    message=_upstream_turn_error_message(retry_turn_error),
                    status_code=502,
                    duration_ms=int((time.perf_counter() - retry_started_at) * 1000),
                    extra_fields={
                        "errorCode": retry_turn_error.error_code or None,
                        "retryReason": "upstream_turn_error",
                    },
                )
                return _native_error_response(
                    502,
                    _upstream_turn_error_message(retry_turn_error),
                )
            _record_attempt(
                stream_account,
                stream_quota,
                retry_request_id,
                attempt=2,
                success=True,
                stream=True,
                stop_reason="upstream_request_completed",
                message="原生 Gemini 上游请求完成",
                status_code=200,
                duration_ms=int((time.perf_counter() - retry_started_at) * 1000),
                extra_fields={"retryReason": "upstream_turn_error"},
            )
        else:
            _record_attempt(
                stream_account,
                stream_quota,
                request_id,
                attempt=1,
                success=True,
                stream=True,
                stop_reason="upstream_request_completed",
                message="原生 Gemini 上游请求完成",
                status_code=200,
                duration_ms=int((time.perf_counter() - attempt_started_at) * 1000),
            )

        response_headers = {
            "x-accio-account-id": stream_account.id,
            "x-accio-account-strategy": panel_settings.api_account_strategy,
        }
        if stream_quota.get("success"):
            response_headers["x-accio-account-remaining"] = str(
                stream_quota["remaining_value"]
            )

        content_type = "text/event-stream"
        upstream_headers = getattr(stream_response, "headers", None)
        if isinstance(upstream_headers, dict):
            candidate = str(upstream_headers.get("content-type") or "").strip()
            if candidate:
                content_type = candidate

        return StreamingResponse(
            itertools.chain(prefetched_chunks, remaining_chunks),
            media_type=content_type,
            headers=response_headers,
        )
