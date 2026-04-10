from __future__ import annotations

import asyncio
import itertools
import json
import time
from typing import Any, Iterator

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..api_logs import ApiLogStore
from ..anthropic_proxy import (
    DEFAULT_ANTHROPIC_MODEL,
    UpstreamTurnError,
    decode_non_stream_response,
)
from ..openai_proxy import (
    build_accio_request_from_openai,
    build_openai_chat_completion_response,
    build_openai_chat_payload_from_responses,
    build_openai_responses_response,
    iter_openai_chat_sse_bytes,
    iter_openai_responses_sse_bytes,
)
from ..upstream_support import (
    extract_upstream_turn_error_from_chunk as _extract_upstream_turn_error_from_chunk,
    is_stream_summary_empty as _is_stream_summary_empty,
    make_upstream_attempt_logger as _make_upstream_attempt_logger,
    openai_chat_chunk_has_meaningful_output as _openai_chat_chunk_has_meaningful_output,
    openai_responses_chunk_has_meaningful_output as _openai_responses_chunk_has_meaningful_output,
    prefetch_stream_until_meaningful as _prefetch_stream_until_meaningful,
    request_upstream_or_error,
    should_retry_upstream_turn_error as _should_retry_upstream_turn_error,
    summarize_non_stream_payload as _summarize_non_stream_payload,
    upstream_turn_error_message as _upstream_turn_error_message,
)
from ..usage_stats import UsageStatsStore
from .context import ProxyRouteContext


def install_openai_routes(context: ProxyRouteContext) -> None:
    application = context.application
    settings = context.settings
    store = context.store
    client = context.client
    panel_settings_store = context.panel_settings_store
    usage_stats_store = context.usage_stats_store
    api_log_store = context.api_log_store
    ProxySelectionError = context.ProxySelectionError

    _extract_proxy_api_key = context.extract_proxy_api_key
    _is_allowed_dynamic_model = context.is_allowed_dynamic_model
    _select_proxy_account = context.select_proxy_account
    _empty_response_log_message = context.empty_response_log_message
    _should_disable_model_on_empty_response = context.should_disable_model_on_empty_response
    _disable_account_model_on_empty_response = context.disable_account_model_on_empty_response
    _openai_error_response = context.openai_error_response

    @application.post("/v1/responses")
    async def openai_responses(request: Request) -> Response:
        panel_settings = panel_settings_store.load()
        usage_stats_store: UsageStatsStore = application.state.usage_stats_store
        api_log_store: ApiLogStore = application.state.api_log_store
        api_key = _extract_proxy_api_key(request)
        if api_key != panel_settings.admin_password:
            return _openai_error_response(
                401,
                "无效的 API Key，请使用管理员密码作为 x-api-key 或 Bearer Token。",
                error_type="authentication_error",
                code="invalid_api_key",
            )

        started_at = time.perf_counter()
        raw_body = await request.body()
        if not raw_body.strip():
            return _openai_error_response(400, "请求体不能为空。")

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _openai_error_response(400, "请求体必须是合法的 JSON。")
        if not isinstance(payload, dict):
            return _openai_error_response(400, "请求体必须是 JSON 对象。")
        model = str(payload.get("model") or DEFAULT_ANTHROPIC_MODEL)
        disable_on_empty_response = _should_disable_model_on_empty_response(
            payload,
            model,
        )
        requested_stream = bool(payload.get("stream", False))
        allowed, available = _is_allowed_dynamic_model(
            application,
            panel_settings,
            model,
        )
        if not allowed:
            return _openai_error_response(
                400,
                f"不支持模型 {model}。当前可用模型: {', '.join(available)}",
                code="model_not_found",
            )

        chat_payload = build_openai_chat_payload_from_responses(payload)
        messages_value = chat_payload.get("messages")
        messages_count = len(messages_value) if isinstance(messages_value, list) else 0
        try:
            max_tokens = int(
                chat_payload.get("max_tokens") or payload.get("max_output_tokens") or 0
            )
        except (TypeError, ValueError):
            max_tokens = 0

        try:
            account, quota = await asyncio.to_thread(
                _select_proxy_account,
                application,
                panel_settings,
                model,
            )
        except ProxySelectionError as exc:
            api_log_store.record(
                {
                    "level": "error",
                    "event": "v1_responses",
                    "success": False,
                    "emptyResponse": False,
                    "accountId": "",
                    "accountName": "-",
                    "fillPriority": None,
                    "model": model,
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
                    "maxTokens": max_tokens,
                    "durationMs": int((time.perf_counter() - started_at) * 1000),
                }
            )
            return _openai_error_response(
                exc.status_code,
                exc.message,
                error_type="server_error",
                code="proxy_selection_failed",
            )

        accio_body = build_accio_request_from_openai(
            chat_payload,
            token=account.access_token,
            utdid=account.utdid,
            version=settings.version,
        )
        request_id = str(accio_body.get("request_id") or "")
        root_request_id = request_id

        _record_attempt = _make_upstream_attempt_logger(
            api_log_store,
            event="v1_responses",
            model=model,
            strategy=panel_settings.api_account_strategy,
            root_request_id=root_request_id,
            messages_count=messages_count,
            max_tokens=max_tokens,
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
                event="v1_responses",
                model=model,
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
                max_tokens=max_tokens,
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
                return _openai_error_response(
                    status_code,
                    message,
                    error_type="server_error",
                    code=(
                        "upstream_request_failed"
                        if stop_reason == "request_exception"
                        else "upstream_error"
                    ),
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
                model=model,
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
                headers = {
                    "x-accio-account-id": selected_account.id,
                    "x-accio-account-strategy": panel_settings.api_account_strategy,
                }
                if selected_quota.get("success"):
                    headers["x-accio-account-remaining"] = str(
                        selected_quota["remaining_value"]
                    )
                return headers

            def _response_meta(
                selected_account: Account,
                selected_quota: dict[str, Any],
                selected_request_id: str,
            ) -> dict[str, Any]:
                return {
                    "account_id": selected_account.id,
                    "account_name": selected_account.name,
                    "fill_priority": selected_account.fill_priority,
                    "strategy": panel_settings.api_account_strategy,
                    "remaining_quota": selected_quota.get("remaining_value"),
                    "used_quota": selected_quota.get("used_value"),
                    "request_id": selected_request_id,
                    "session_id": chat_payload.get("session_id"),
                    "conversation_id": chat_payload.get("conversation_id"),
                    "previous_response_id": chat_payload.get("previous_response_id"),
                }

            def _build_stream_attempt(
                selected_account: Account,
                selected_quota: dict[str, Any],
                selected_response: requests.Response,
                selected_request_id: str,
                selected_attempt: int,
                selected_attempt_started_at: float,
            ) -> tuple[Iterator[bytes], bool]:
                def on_openai_responses_complete(summary: dict[str, Any]) -> None:
                    usage = summary.get("usage") if isinstance(summary.get("usage"), dict) else {}
                    empty_response = _is_stream_summary_empty(summary)
                    if empty_response and disable_on_empty_response:
                        _disable_account_model_on_empty_response(
                            store,
                            selected_account,
                            model,
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
                            else str(summary.get("stop_reason") or "end_turn")
                        ),
                        message=(
                            _empty_response_log_message(
                                model,
                                disable_model=disable_on_empty_response,
                            )
                            if empty_response
                            else "Responses 上游流式请求完成"
                        ),
                        status_code=200,
                        input_tokens=int(usage.get("input_tokens") or 0),
                        output_tokens=int(usage.get("output_tokens") or 0),
                        empty_response=empty_response,
                        duration_ms=int(
                            (time.perf_counter() - selected_attempt_started_at) * 1000
                        ),
                        level="warn" if empty_response else None,
                        extra_fields={
                            "textChars": int(summary.get("text_chars") or 0),
                            "toolUseBlocks": int(summary.get("tool_use_blocks") or 0),
                        },
                    )
                    usage_stats_store.record_message(
                        account_id=selected_account.id,
                        model=model,
                        input_tokens=int(usage.get("input_tokens") or 0),
                        output_tokens=int(usage.get("output_tokens") or 0),
                        success=True,
                        stop_reason=str(summary.get("stop_reason") or "end_turn"),
                    )
                    api_log_store.record(
                        {
                            "level": "warn" if empty_response else "info",
                            "event": "v1_responses",
                            "success": True,
                            "emptyResponse": empty_response,
                            "accountId": selected_account.id,
                            "accountName": selected_account.name,
                            "fillPriority": selected_account.fill_priority,
                            "model": model,
                            "stream": True,
                            "strategy": panel_settings.api_account_strategy,
                            "requestId": selected_request_id,
                            "message": (
                                _empty_response_log_message(
                                    model,
                                    disable_model=disable_on_empty_response,
                                )
                                if empty_response
                                else "Responses 流式调用完成"
                            ),
                            "statusCode": 200,
                            "stopReason": str(summary.get("stop_reason") or "end_turn"),
                            "inputTokens": int(usage.get("input_tokens") or 0),
                            "outputTokens": int(usage.get("output_tokens") or 0),
                            "remainingQuota": selected_quota.get("remaining_value"),
                            "usedQuota": selected_quota.get("used_value"),
                            "messagesCount": messages_count,
                            "maxTokens": max_tokens,
                            "textChars": int(summary.get("text_chars") or 0),
                            "toolUseBlocks": int(summary.get("tool_use_blocks") or 0),
                            "durationMs": int((time.perf_counter() - started_at) * 1000),
                        }
                    )

                stream_bytes = iter_openai_responses_sse_bytes(
                    selected_response,
                    model,
                    accio=_response_meta(
                        selected_account,
                        selected_quota,
                        selected_request_id,
                    ),
                    on_complete=on_openai_responses_complete,
                )
                prefetched_chunks, remaining_chunks, has_meaningful_output = (
                    _prefetch_stream_until_meaningful(
                        stream_bytes,
                        chunk_has_meaningful_output=_openai_responses_chunk_has_meaningful_output,
                    )
                )
                return (
                    itertools.chain(prefetched_chunks, remaining_chunks),
                    has_meaningful_output,
                )

            stream_account = account
            stream_quota = quota
            stream_request_id = request_id
            try:
                stream_bytes, has_meaningful_output = _build_stream_attempt(
                    stream_account,
                    stream_quota,
                    upstream_response,
                    stream_request_id,
                    1,
                    attempt_started_at,
                )
            except UpstreamTurnError as exc:
                _record_attempt(
                    stream_account,
                    stream_quota,
                    stream_request_id,
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
                    return _openai_error_response(
                        502,
                        _upstream_turn_error_message(exc),
                        error_type="server_error",
                        code="upstream_error",
                    )
                has_meaningful_output = False
            if not has_meaningful_output:
                try:
                    retry_account, retry_quota = await asyncio.to_thread(
                        _select_proxy_account,
                        application,
                        panel_settings,
                        model,
                        exclude_account_ids={stream_account.id},
                    )
                except ProxySelectionError as exc:
                    return _openai_error_response(
                        exc.status_code,
                        exc.message,
                        error_type="server_error",
                        code="proxy_selection_failed",
                    )

                retry_body = build_accio_request_from_openai(
                    chat_payload,
                    token=retry_account.access_token,
                    utdid=retry_account.utdid,
                    version=settings.version,
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
                stream_request_id = retry_request_id
                try:
                    stream_bytes, _ = _build_stream_attempt(
                        stream_account,
                        stream_quota,
                        retry_response,
                        stream_request_id,
                        2,
                        retry_started_at,
                    )
                except UpstreamTurnError as exc:
                    _record_attempt(
                        stream_account,
                        stream_quota,
                        stream_request_id,
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
                    return _openai_error_response(
                        502,
                        _upstream_turn_error_message(exc),
                        error_type="server_error",
                        code="upstream_error",
                    )

            return StreamingResponse(
                stream_bytes,
                media_type="text/event-stream",
                headers=_stream_headers(stream_account, stream_quota),
            )

        should_retry = False
        retry_due_to_upstream_turn_error = False
        current_attempt = 1
        current_attempt_started_at = attempt_started_at
        try:
            response_payload = await asyncio.to_thread(
                decode_non_stream_response,
                upstream_response,
                model,
            )
        except UpstreamTurnError as exc:
            _record_attempt(
                account,
                quota,
                request_id,
                attempt=1,
                stream=False,
                success=False,
                stop_reason="upstream_turn_error",
                message=_upstream_turn_error_message(exc),
                status_code=502,
                duration_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                extra_fields={"errorCode": exc.error_code or None},
            )
            if not _should_retry_upstream_turn_error(exc):
                return _openai_error_response(
                    502,
                    _upstream_turn_error_message(exc),
                    error_type="server_error",
                    code="upstream_error",
                )
            should_retry = True
            retry_due_to_upstream_turn_error = True
        else:
            usage = response_payload.get("usage") if isinstance(response_payload, dict) else {}
            if not isinstance(usage, dict):
                usage = {}
            output_summary = _summarize_non_stream_payload(response_payload)
            should_retry = bool(output_summary["empty_response"])
        if should_retry:
            if not retry_due_to_upstream_turn_error:
                if disable_on_empty_response:
                    _disable_account_model_on_empty_response(
                        store,
                        account,
                        model,
                    )
                usage_stats_store.record_message(
                    account_id=account.id,
                    model=model,
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    success=True,
                    stop_reason=str(response_payload.get("stop_reason") or "end_turn"),
                )
                _record_attempt(
                    account,
                    quota,
                    request_id,
                    attempt=1,
                    stream=False,
                    success=False,
                    stop_reason="empty_response",
                    message=_empty_response_log_message(
                        model,
                        disable_model=disable_on_empty_response,
                    ),
                    status_code=200,
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    empty_response=True,
                    duration_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                    level="warn",
                    extra_fields={
                        "textChars": int(output_summary["text_chars"]),
                        "toolUseBlocks": int(output_summary["tool_use_blocks"]),
                    },
                )
                api_log_store.record(
                    {
                        "level": "warn",
                        "event": "v1_responses",
                        "success": True,
                        "emptyResponse": True,
                        "accountId": account.id,
                        "accountName": account.name,
                        "fillPriority": account.fill_priority,
                        "model": model,
                        "stream": False,
                        "strategy": panel_settings.api_account_strategy,
                        "requestId": request_id,
                        "message": _empty_response_log_message(
                            model,
                            disable_model=disable_on_empty_response,
                        ),
                        "statusCode": 200,
                        "stopReason": str(response_payload.get("stop_reason") or "end_turn"),
                        "inputTokens": int(usage.get("input_tokens") or 0),
                        "outputTokens": int(usage.get("output_tokens") or 0),
                        "remainingQuota": quota.get("remaining_value"),
                        "usedQuota": quota.get("used_value"),
                        "messagesCount": messages_count,
                        "maxTokens": max_tokens,
                        "textChars": int(output_summary["text_chars"]),
                        "toolUseBlocks": int(output_summary["tool_use_blocks"]),
                        "durationMs": int((time.perf_counter() - started_at) * 1000),
                    }
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
                return _openai_error_response(
                    exc.status_code,
                    exc.message,
                    error_type="server_error",
                    code="proxy_selection_failed",
                )

            retry_body = build_accio_request_from_openai(
                chat_payload,
                token=retry_account.access_token,
                utdid=retry_account.utdid,
                version=settings.version,
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
                retry_reason=(
                    "upstream_turn_error"
                    if retry_due_to_upstream_turn_error
                    else "empty_response"
                ),
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
                    decode_non_stream_response,
                    retry_response,
                    model,
                )
            except UpstreamTurnError as exc:
                _record_attempt(
                    account,
                    quota,
                    request_id,
                    attempt=2,
                    stream=False,
                    success=False,
                    stop_reason="upstream_turn_error",
                    message=_upstream_turn_error_message(exc),
                    status_code=502,
                    duration_ms=int((time.perf_counter() - retry_started_at) * 1000),
                    extra_fields={
                        "errorCode": exc.error_code or None,
                        "retryReason": (
                            "upstream_turn_error"
                            if retry_due_to_upstream_turn_error
                            else "empty_response"
                        ),
                    },
                )
                return _openai_error_response(
                    502,
                    _upstream_turn_error_message(exc),
                    error_type="server_error",
                    code="upstream_error",
                )
            usage = (
                response_payload.get("usage")
                if isinstance(response_payload, dict)
                else {}
            )
            if not isinstance(usage, dict):
                usage = {}
            output_summary = _summarize_non_stream_payload(response_payload)
            if output_summary["empty_response"] and disable_on_empty_response:
                _disable_account_model_on_empty_response(
                    store,
                    account,
                    model,
                )

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
                else str(response_payload.get("stop_reason") or "end_turn")
            ),
            message=(
                _empty_response_log_message(
                    model,
                    disable_model=disable_on_empty_response,
                )
                if output_summary["empty_response"]
                else "Responses 上游请求完成"
            ),
            status_code=200,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            empty_response=bool(output_summary["empty_response"]),
            duration_ms=int((time.perf_counter() - current_attempt_started_at) * 1000),
            level="warn" if output_summary["empty_response"] else None,
            extra_fields={
                "textChars": int(output_summary["text_chars"]),
                "toolUseBlocks": int(output_summary["tool_use_blocks"]),
                "retryReason": (
                    None
                    if current_attempt == 1
                    else (
                        "upstream_turn_error"
                        if retry_due_to_upstream_turn_error
                        else "empty_response"
                    )
                ),
            },
        )
        response_headers = {
            "x-accio-account-id": account.id,
            "x-accio-account-strategy": panel_settings.api_account_strategy,
        }
        if quota.get("success"):
            response_headers["x-accio-account-remaining"] = str(
                quota["remaining_value"]
            )
        accio_response_meta = {
            "account_id": account.id,
            "account_name": account.name,
            "fill_priority": account.fill_priority,
            "strategy": panel_settings.api_account_strategy,
            "remaining_quota": quota.get("remaining_value"),
            "used_quota": quota.get("used_value"),
            "request_id": request_id,
            "session_id": chat_payload.get("session_id"),
            "conversation_id": chat_payload.get("conversation_id"),
            "previous_response_id": chat_payload.get("previous_response_id"),
        }
        usage_stats_store.record_message(
            account_id=account.id,
            model=model,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            success=True,
            stop_reason=str(response_payload.get("stop_reason") or "end_turn"),
        )
        api_log_store.record(
            {
                "level": "warn" if output_summary["empty_response"] else "info",
                "event": "v1_responses",
                "success": True,
                "emptyResponse": bool(output_summary["empty_response"]),
                "accountId": account.id,
                "accountName": account.name,
                "fillPriority": account.fill_priority,
                "model": model,
                "stream": False,
                "strategy": panel_settings.api_account_strategy,
                "requestId": request_id,
                "message": (
                    _empty_response_log_message(
                        model,
                        disable_model=disable_on_empty_response,
                    )
                    if output_summary["empty_response"]
                    else "Responses 非流式调用完成"
                ),
                "statusCode": 200,
                "stopReason": str(response_payload.get("stop_reason") or "end_turn"),
                "inputTokens": int(usage.get("input_tokens") or 0),
                "outputTokens": int(usage.get("output_tokens") or 0),
                "remainingQuota": quota.get("remaining_value"),
                "usedQuota": quota.get("used_value"),
                "messagesCount": messages_count,
                "maxTokens": max_tokens,
                "textChars": int(output_summary["text_chars"]),
                "toolUseBlocks": int(output_summary["tool_use_blocks"]),
                "durationMs": int((time.perf_counter() - started_at) * 1000),
            }
        )
        return JSONResponse(
            build_openai_responses_response(
                response_payload,
                model=model,
                accio=accio_response_meta,
            ),
            headers=response_headers,
        )

    @application.post("/v1/chat/completions")
    @application.post("/chat/completions")
    async def openai_chat_completions(request: Request) -> Response:
        panel_settings = panel_settings_store.load()
        usage_stats_store: UsageStatsStore = application.state.usage_stats_store
        api_log_store: ApiLogStore = application.state.api_log_store
        api_key = _extract_proxy_api_key(request)
        if api_key != panel_settings.admin_password:
            return _openai_error_response(
                401,
                "无效的 API Key，请使用管理员密码作为 x-api-key 或 Bearer Token。",
                error_type="authentication_error",
                code="invalid_api_key",
            )

        started_at = time.perf_counter()
        raw_body = await request.body()
        if not raw_body.strip():
            return _openai_error_response(400, "请求体不能为空。")

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _openai_error_response(400, "请求体必须是合法的 JSON。")
        if not isinstance(payload, dict):
            return _openai_error_response(400, "请求体必须是 JSON 对象。")

        model = str(payload.get("model") or DEFAULT_ANTHROPIC_MODEL)
        disable_on_empty_response = _should_disable_model_on_empty_response(
            payload,
            model,
        )
        requested_stream = bool(payload.get("stream", False))
        messages_value = payload.get("messages")
        messages_count = len(messages_value) if isinstance(messages_value, list) else 0
        try:
            max_tokens = int(
                payload.get("max_completion_tokens") or payload.get("max_tokens") or 0
            )
        except (TypeError, ValueError):
            max_tokens = 0
        allowed, available = _is_allowed_dynamic_model(
            application,
            panel_settings,
            model,
        )
        if not allowed:
            return _openai_error_response(
                400,
                f"不支持模型 {model}。当前可用模型: {', '.join(available)}",
                code="model_not_found",
            )

        try:
            account, quota = await asyncio.to_thread(
                _select_proxy_account,
                application,
                panel_settings,
                model,
            )
        except ProxySelectionError as exc:
            api_log_store.record(
                {
                    "level": "error",
                    "event": "v1_chat_completions",
                    "success": False,
                    "emptyResponse": False,
                    "accountId": "",
                    "accountName": "-",
                    "fillPriority": None,
                    "model": model,
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
                    "maxTokens": max_tokens,
                    "durationMs": int((time.perf_counter() - started_at) * 1000),
                }
            )
            return _openai_error_response(
                exc.status_code,
                exc.message,
                error_type="server_error",
                code="proxy_selection_failed",
            )

        accio_body = build_accio_request_from_openai(
            payload,
            token=account.access_token,
            utdid=account.utdid,
            version=settings.version,
        )
        request_id = str(accio_body.get("request_id") or "")
        root_request_id = request_id

        _record_attempt = _make_upstream_attempt_logger(
            api_log_store,
            event="v1_chat_completions",
            model=model,
            strategy=panel_settings.api_account_strategy,
            root_request_id=root_request_id,
            messages_count=messages_count,
            max_tokens=max_tokens,
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
                event="v1_chat_completions",
                model=model,
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
                max_tokens=max_tokens,
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
                return _openai_error_response(
                    status_code,
                    message,
                    error_type="server_error",
                    code=(
                        "upstream_request_failed"
                        if stop_reason == "request_exception"
                        else "upstream_error"
                    ),
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
                model=model,
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
                headers = {
                    "x-accio-account-id": selected_account.id,
                    "x-accio-account-strategy": panel_settings.api_account_strategy,
                }
                if selected_quota.get("success"):
                    headers["x-accio-account-remaining"] = str(
                        selected_quota["remaining_value"]
                    )
                return headers

            def _build_stream_attempt(
                selected_account: Account,
                selected_quota: dict[str, Any],
                selected_response: requests.Response,
                selected_request_id: str,
                selected_attempt: int,
                selected_attempt_started_at: float,
            ) -> tuple[Iterator[bytes], bool]:
                def on_openai_stream_complete(summary: dict[str, Any]) -> None:
                    usage = summary.get("usage") if isinstance(summary.get("usage"), dict) else {}
                    empty_response = _is_stream_summary_empty(summary)
                    if empty_response and disable_on_empty_response:
                        _disable_account_model_on_empty_response(
                            store,
                            selected_account,
                            model,
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
                            else str(summary.get("stop_reason") or "end_turn")
                        ),
                        message=(
                            _empty_response_log_message(
                                model,
                                disable_model=disable_on_empty_response,
                            )
                            if empty_response
                            else "OpenAI chat 上游流式请求完成"
                        ),
                        status_code=200,
                        input_tokens=int(usage.get("input_tokens") or 0),
                        output_tokens=int(usage.get("output_tokens") or 0),
                        empty_response=empty_response,
                        duration_ms=int(
                            (time.perf_counter() - selected_attempt_started_at) * 1000
                        ),
                        level="warn" if empty_response else None,
                        extra_fields={
                            "textChars": int(summary.get("text_chars") or 0),
                            "toolUseBlocks": int(summary.get("tool_use_blocks") or 0),
                        },
                    )
                    usage_stats_store.record_message(
                        account_id=selected_account.id,
                        model=model,
                        input_tokens=int(usage.get("input_tokens") or 0),
                        output_tokens=int(usage.get("output_tokens") or 0),
                        success=True,
                        stop_reason=str(summary.get("stop_reason") or "end_turn"),
                    )
                    api_log_store.record(
                        {
                            "level": "warn" if empty_response else "info",
                            "event": "v1_chat_completions",
                            "success": True,
                            "emptyResponse": empty_response,
                            "accountId": selected_account.id,
                            "accountName": selected_account.name,
                            "fillPriority": selected_account.fill_priority,
                            "model": model,
                            "stream": True,
                            "strategy": panel_settings.api_account_strategy,
                            "requestId": selected_request_id,
                            "message": (
                                _empty_response_log_message(
                                    model,
                                    disable_model=disable_on_empty_response,
                                )
                                if empty_response
                                else "OpenAI 流式调用完成"
                            ),
                            "statusCode": 200,
                            "stopReason": str(summary.get("stop_reason") or "end_turn"),
                            "inputTokens": int(usage.get("input_tokens") or 0),
                            "outputTokens": int(usage.get("output_tokens") or 0),
                            "remainingQuota": selected_quota.get("remaining_value"),
                            "usedQuota": selected_quota.get("used_value"),
                            "messagesCount": messages_count,
                            "maxTokens": max_tokens,
                            "textChars": int(summary.get("text_chars") or 0),
                            "toolUseBlocks": int(summary.get("tool_use_blocks") or 0),
                            "durationMs": int((time.perf_counter() - started_at) * 1000),
                        }
                    )

                stream_bytes = iter_openai_chat_sse_bytes(
                    selected_response,
                    model,
                    on_complete=on_openai_stream_complete,
                )
                prefetched_chunks, remaining_chunks, has_meaningful_output = (
                    _prefetch_stream_until_meaningful(
                        stream_bytes,
                        chunk_has_meaningful_output=_openai_chat_chunk_has_meaningful_output,
                    )
                )
                return (
                    itertools.chain(prefetched_chunks, remaining_chunks),
                    has_meaningful_output,
                )

            stream_account = account
            stream_quota = quota
            stream_request_id = request_id
            try:
                stream_bytes, has_meaningful_output = _build_stream_attempt(
                    stream_account,
                    stream_quota,
                    upstream_response,
                    stream_request_id,
                    1,
                    attempt_started_at,
                )
            except UpstreamTurnError as exc:
                _record_attempt(
                    stream_account,
                    stream_quota,
                    stream_request_id,
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
                    return _openai_error_response(
                        502,
                        _upstream_turn_error_message(exc),
                        error_type="server_error",
                        code="upstream_error",
                    )
                has_meaningful_output = False
            if not has_meaningful_output:
                try:
                    retry_account, retry_quota = await asyncio.to_thread(
                        _select_proxy_account,
                        application,
                        panel_settings,
                        model,
                        exclude_account_ids={stream_account.id},
                    )
                except ProxySelectionError as exc:
                    return _openai_error_response(
                        exc.status_code,
                        exc.message,
                        error_type="server_error",
                        code="proxy_selection_failed",
                    )

                retry_body = build_accio_request_from_openai(
                    payload,
                    token=retry_account.access_token,
                    utdid=retry_account.utdid,
                    version=settings.version,
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
                stream_request_id = retry_request_id
                try:
                    stream_bytes, _ = _build_stream_attempt(
                        stream_account,
                        stream_quota,
                        retry_response,
                        stream_request_id,
                        2,
                        retry_started_at,
                    )
                except UpstreamTurnError as exc:
                    _record_attempt(
                        stream_account,
                        stream_quota,
                        stream_request_id,
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
                    return _openai_error_response(
                        502,
                        _upstream_turn_error_message(exc),
                        error_type="server_error",
                        code="upstream_error",
                    )

            return StreamingResponse(
                stream_bytes,
                media_type="text/event-stream",
                headers=_stream_headers(stream_account, stream_quota),
            )

        should_retry = False
        retry_due_to_upstream_turn_error = False
        current_attempt = 1
        current_attempt_started_at = attempt_started_at
        try:
            response_payload = await asyncio.to_thread(
                decode_non_stream_response,
                upstream_response,
                model,
            )
        except UpstreamTurnError as exc:
            _record_attempt(
                account,
                quota,
                request_id,
                attempt=1,
                stream=False,
                success=False,
                stop_reason="upstream_turn_error",
                message=_upstream_turn_error_message(exc),
                status_code=502,
                duration_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                extra_fields={"errorCode": exc.error_code or None},
            )
            if not _should_retry_upstream_turn_error(exc):
                return _openai_error_response(
                    502,
                    _upstream_turn_error_message(exc),
                    error_type="server_error",
                    code="upstream_error",
                )
            should_retry = True
            retry_due_to_upstream_turn_error = True
        else:
            usage = response_payload.get("usage") if isinstance(response_payload, dict) else {}
            if not isinstance(usage, dict):
                usage = {}
            output_summary = _summarize_non_stream_payload(response_payload)
            should_retry = bool(output_summary["empty_response"])
        if should_retry:
            if not retry_due_to_upstream_turn_error:
                if disable_on_empty_response:
                    _disable_account_model_on_empty_response(
                        store,
                        account,
                        model,
                    )
                usage_stats_store.record_message(
                    account_id=account.id,
                    model=model,
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    success=True,
                    stop_reason=str(response_payload.get("stop_reason") or "end_turn"),
                )
                _record_attempt(
                    account,
                    quota,
                    request_id,
                    attempt=1,
                    stream=False,
                    success=False,
                    stop_reason="empty_response",
                    message=_empty_response_log_message(
                        model,
                        disable_model=disable_on_empty_response,
                    ),
                    status_code=200,
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    empty_response=True,
                    duration_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                    level="warn",
                    extra_fields={
                        "textChars": int(output_summary["text_chars"]),
                        "toolUseBlocks": int(output_summary["tool_use_blocks"]),
                    },
                )
                api_log_store.record(
                    {
                        "level": "warn",
                        "event": "v1_chat_completions",
                        "success": True,
                        "emptyResponse": True,
                        "accountId": account.id,
                        "accountName": account.name,
                        "fillPriority": account.fill_priority,
                        "model": model,
                        "stream": False,
                        "strategy": panel_settings.api_account_strategy,
                        "requestId": request_id,
                        "message": _empty_response_log_message(
                            model,
                            disable_model=disable_on_empty_response,
                        ),
                        "statusCode": 200,
                        "stopReason": str(response_payload.get("stop_reason") or "end_turn"),
                        "inputTokens": int(usage.get("input_tokens") or 0),
                        "outputTokens": int(usage.get("output_tokens") or 0),
                        "remainingQuota": quota.get("remaining_value"),
                        "usedQuota": quota.get("used_value"),
                        "messagesCount": messages_count,
                        "maxTokens": max_tokens,
                        "textChars": int(output_summary["text_chars"]),
                        "toolUseBlocks": int(output_summary["tool_use_blocks"]),
                        "durationMs": int((time.perf_counter() - started_at) * 1000),
                    }
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
                return _openai_error_response(
                    exc.status_code,
                    exc.message,
                    error_type="server_error",
                    code="proxy_selection_failed",
                )

            retry_body = build_accio_request_from_openai(
                payload,
                token=retry_account.access_token,
                utdid=retry_account.utdid,
                version=settings.version,
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
                retry_reason=(
                    "upstream_turn_error"
                    if retry_due_to_upstream_turn_error
                    else "empty_response"
                ),
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
                    decode_non_stream_response,
                    retry_response,
                    model,
                )
            except UpstreamTurnError as exc:
                _record_attempt(
                    account,
                    quota,
                    request_id,
                    attempt=2,
                    stream=False,
                    success=False,
                    stop_reason="upstream_turn_error",
                    message=_upstream_turn_error_message(exc),
                    status_code=502,
                    duration_ms=int((time.perf_counter() - retry_started_at) * 1000),
                    extra_fields={
                        "errorCode": exc.error_code or None,
                        "retryReason": (
                            "upstream_turn_error"
                            if retry_due_to_upstream_turn_error
                            else "empty_response"
                        ),
                    },
                )
                return _openai_error_response(
                    502,
                    _upstream_turn_error_message(exc),
                    error_type="server_error",
                    code="upstream_error",
                )
            usage = (
                response_payload.get("usage")
                if isinstance(response_payload, dict)
                else {}
            )
            if not isinstance(usage, dict):
                usage = {}
            output_summary = _summarize_non_stream_payload(response_payload)
            if output_summary["empty_response"] and disable_on_empty_response:
                _disable_account_model_on_empty_response(
                    store,
                    account,
                    model,
                )

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
                else str(response_payload.get("stop_reason") or "end_turn")
            ),
            message=(
                _empty_response_log_message(
                    model,
                    disable_model=disable_on_empty_response,
                )
                if output_summary["empty_response"]
                else "OpenAI chat 上游请求完成"
            ),
            status_code=200,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            empty_response=bool(output_summary["empty_response"]),
            duration_ms=int((time.perf_counter() - current_attempt_started_at) * 1000),
            level="warn" if output_summary["empty_response"] else None,
            extra_fields={
                "textChars": int(output_summary["text_chars"]),
                "toolUseBlocks": int(output_summary["tool_use_blocks"]),
                "retryReason": (
                    None
                    if current_attempt == 1
                    else (
                        "upstream_turn_error"
                        if retry_due_to_upstream_turn_error
                        else "empty_response"
                    )
                ),
            },
        )
        response_headers = {
            "x-accio-account-id": account.id,
            "x-accio-account-strategy": panel_settings.api_account_strategy,
        }
        if quota.get("success"):
            response_headers["x-accio-account-remaining"] = str(
                quota["remaining_value"]
            )
        accio_chat_meta = {
            "account_id": account.id,
            "account_name": account.name,
            "fill_priority": account.fill_priority,
            "strategy": panel_settings.api_account_strategy,
            "remaining_quota": quota.get("remaining_value"),
            "used_quota": quota.get("used_value"),
            "request_id": request_id,
            "session_id": payload.get("session_id", payload.get("sessionId")),
            "conversation_id": payload.get(
                "conversation_id",
                payload.get("conversationId"),
            ),
        }
        usage_stats_store.record_message(
            account_id=account.id,
            model=model,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            success=True,
            stop_reason=str(response_payload.get("stop_reason") or "end_turn"),
        )
        api_log_store.record(
            {
                "level": "warn" if output_summary["empty_response"] else "info",
                "event": "v1_chat_completions",
                "success": True,
                "emptyResponse": bool(output_summary["empty_response"]),
                "accountId": account.id,
                "accountName": account.name,
                "fillPriority": account.fill_priority,
                "model": model,
                "stream": False,
                "strategy": panel_settings.api_account_strategy,
                "requestId": request_id,
                "message": (
                    _empty_response_log_message(
                        model,
                        disable_model=disable_on_empty_response,
                    )
                    if output_summary["empty_response"]
                    else "OpenAI 非流式调用完成"
                ),
                "statusCode": 200,
                "stopReason": str(response_payload.get("stop_reason") or "end_turn"),
                "inputTokens": int(usage.get("input_tokens") or 0),
                "outputTokens": int(usage.get("output_tokens") or 0),
                "remainingQuota": quota.get("remaining_value"),
                "usedQuota": quota.get("used_value"),
                "messagesCount": messages_count,
                "maxTokens": max_tokens,
                "textChars": int(output_summary["text_chars"]),
                "toolUseBlocks": int(output_summary["tool_use_blocks"]),
                "durationMs": int((time.perf_counter() - started_at) * 1000),
            }
        )
        return JSONResponse(
            build_openai_chat_completion_response(
                response_payload,
                model=model,
                accio=accio_chat_meta,
            ),
            headers=response_headers,
        )
