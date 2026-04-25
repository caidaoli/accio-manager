from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..api_logs import ApiLogStore
from ..anthropic_proxy import (
    DEFAULT_ANTHROPIC_MODEL,
    UpstreamTurnError,
    build_accio_request,
    decode_non_stream_response,
    iter_anthropic_sse_bytes,
)
from ..upstream_support import (
    anthropic_stream_chunk_has_meaningful_output as _anthropic_stream_chunk_has_meaningful_output,
    extract_upstream_turn_error_from_chunk as _extract_upstream_turn_error_from_chunk,
    is_retryable_quota_exhausted_turn_error as _is_retryable_quota_exhausted_turn_error,
    make_upstream_attempt_logger as _make_upstream_attempt_logger,
    request_upstream_or_error,
    should_retry_upstream_turn_error as _should_retry_upstream_turn_error,
    summarize_non_stream_payload as _summarize_non_stream_payload,
    upstream_turn_error_message as _upstream_turn_error_message,
)
from ..usage_stats import UsageStatsStore
from .context import ProxyRouteContext
from .shared import (
    ProxyEndpointConfig,
    make_build_stream_attempt,
    make_build_upstream_error_response,
    make_record_final_log,
    make_stream_headers,
)


def install_anthropic_routes(context: ProxyRouteContext) -> None:
    application = context.application
    settings = context.settings
    client = context.client
    panel_settings_store = context.panel_settings_store
    usage_stats_store = context.usage_stats_store
    api_log_store = context.api_log_store
    store = context.store
    ProxySelectionError = context.ProxySelectionError

    _extract_proxy_api_key = context.extract_proxy_api_key
    _authorize_proxy_request = context.authorize_proxy_request
    _is_allowed_dynamic_model = context.is_allowed_dynamic_model
    _select_proxy_account = context.select_proxy_account
    _empty_response_log_message = context.empty_response_log_message
    _should_disable_model_on_empty_response = context.should_disable_model_on_empty_response
    _disable_account_model_on_empty_response = context.disable_account_model_on_empty_response
    _mark_account_quota_exhausted_cooldown = context.mark_account_quota_exhausted_cooldown
    _anthropic_error_response = context.anthropic_error_response

    @application.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        panel_settings = panel_settings_store.load()
        usage_stats_store: UsageStatsStore = application.state.usage_stats_store
        api_log_store: ApiLogStore = application.state.api_log_store
        unauthorized = _authorize_proxy_request(request, panel_settings)
        if unauthorized:
            return unauthorized

        started_at = time.perf_counter()

        raw_body = await request.body()
        if not raw_body.strip():
            return _anthropic_error_response(
                400,
                "请求体不能为空。",
                error_type="invalid_request_error",
            )

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _anthropic_error_response(
                400,
                "请求体必须是合法的 JSON。",
                error_type="invalid_request_error",
            )

        if not isinstance(payload, dict):
            return _anthropic_error_response(
                400,
                "请求体必须是 JSON 对象。",
                error_type="invalid_request_error",
            )

        model = str(payload.get("model") or DEFAULT_ANTHROPIC_MODEL)
        disable_on_empty_response = _should_disable_model_on_empty_response(
            payload,
            model,
        )
        requested_stream = bool(payload.get("stream", False))
        messages_value = payload.get("messages")
        messages_count = len(messages_value) if isinstance(messages_value, list) else 0
        try:
            max_tokens = int(payload.get("max_tokens") or 0)
        except (TypeError, ValueError):
            max_tokens = 0
        allowed, available = _is_allowed_dynamic_model(
            application,
            panel_settings,
            model,
        )
        if not allowed:
            return _anthropic_error_response(
                400,
                f"不支持模型 {model}。当前可用模型: {', '.join(available)}",
                error_type="invalid_request_error",
            )

        try:
            account, quota = await asyncio.to_thread(
                _select_proxy_account,
                application,
                panel_settings,
            )
        except ProxySelectionError as exc:
            api_log_store.record(
                {
                    "level": "error",
                    "event": "v1_messages",
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
            return _anthropic_error_response(exc.status_code, exc.message)

        accio_body = build_accio_request(
            payload,
            token=account.access_token,
            utdid=account.utdid,
            version=settings.version,
        )
        request_id = str(accio_body.get("request_id") or "")
        root_request_id = request_id

        _record_attempt = _make_upstream_attempt_logger(
            api_log_store,
            event="v1_messages",
            model=model,
            strategy=panel_settings.api_account_strategy,
            root_request_id=root_request_id,
            messages_count=messages_count,
            max_tokens=max_tokens,
        )

        endpoint_config = ProxyEndpointConfig(
            event="v1_messages",
            model=model,
            default_stop_reason="end_turn",
            stream_complete_message="Anthropic 上游流式请求完成",
            error_response_builder=lambda sc, msg, _sr: _anthropic_error_response(sc, msg),
            extra_fields_extractor=lambda s: {
                "cacheCreationInputTokens": int(s.get("cache_creation_input_tokens") or 0),
                "cacheReadInputTokens": int(s.get("cache_read_input_tokens") or 0),
                "textChars": int(s.get("text_chars") or 0),
                "thinkingChars": int(s.get("thinking_chars") or 0),
                "toolUseBlocks": int(s.get("tool_use_blocks") or 0),
            },
            max_tokens=max_tokens,
            disable_on_empty_response=disable_on_empty_response,
            cache_token_fields=["cache_creation_input_tokens", "cache_read_input_tokens"],
        )

        _record_final_log = make_record_final_log(
            config=endpoint_config,
            api_log_store=api_log_store,
            panel_settings=panel_settings,
            started_at=started_at,
            messages_count=messages_count,
        )

        _build_upstream_error_response = make_build_upstream_error_response(
            config=endpoint_config,
            record_final_log=_record_final_log,
        )

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
            _stream_headers = make_stream_headers(
                panel_settings=panel_settings,
                include_remaining=True,
            )

            _build_stream_attempt = make_build_stream_attempt(
                config=endpoint_config,
                panel_settings=panel_settings,
                store=store,
                usage_stats_store=usage_stats_store,
                api_log_store=api_log_store,
                started_at=started_at,
                messages_count=messages_count,
                record_attempt=_record_attempt,
                disable_account_model_on_empty_response=_disable_account_model_on_empty_response,
                empty_response_log_message=_empty_response_log_message,
                iter_sse_bytes=iter_anthropic_sse_bytes,
                chunk_has_meaningful_output=_anthropic_stream_chunk_has_meaningful_output,
            )

            stream_account = account
            stream_quota = quota
            stream_request_id = request_id
            stream_response = upstream_response
            stream_attempt = 1
            stream_attempt_started_at = attempt_started_at
            stream_retry_reason: str | None = None
            excluded_account_ids = {stream_account.id}
            while True:
                retryable_turn_error = False
                try:
                    stream_bytes, has_meaningful_output = _build_stream_attempt(
                        stream_account,
                        stream_quota,
                        stream_response,
                        stream_request_id,
                        stream_attempt,
                        stream_attempt_started_at,
                    )
                except UpstreamTurnError as exc:
                    extra_fields = {"errorCode": exc.error_code or None}
                    if stream_retry_reason:
                        extra_fields["retryReason"] = stream_retry_reason
                    _record_attempt(
                        stream_account,
                        stream_quota,
                        stream_request_id,
                        attempt=stream_attempt,
                        stream=True,
                        success=False,
                        stop_reason="upstream_turn_error",
                        message=_upstream_turn_error_message(exc),
                        status_code=502,
                        duration_ms=int(
                            (time.perf_counter() - stream_attempt_started_at) * 1000
                        ),
                        extra_fields=extra_fields,
                    )
                    if not _should_retry_upstream_turn_error(exc):
                        return _anthropic_error_response(
                            502,
                            _upstream_turn_error_message(exc),
                        )
                    if _is_retryable_quota_exhausted_turn_error(exc):
                        _mark_account_quota_exhausted_cooldown(store, stream_account)
                    retryable_turn_error = True
                    has_meaningful_output = False
                if has_meaningful_output:
                    break
                if not retryable_turn_error and stream_attempt != 1:
                    break

                try:
                    retry_account, retry_quota = await asyncio.to_thread(
                        _select_proxy_account,
                        application,
                        panel_settings,
                        model,
                        exclude_account_ids=excluded_account_ids,
                    )
                except ProxySelectionError as exc:
                    return _anthropic_error_response(exc.status_code, exc.message)

                retry_body = build_accio_request(
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
                    attempt=stream_attempt + 1,
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
                stream_response = retry_response
                stream_attempt += 1
                stream_attempt_started_at = retry_started_at
                stream_retry_reason = "upstream_turn_error_or_empty_response"
                excluded_account_ids.add(stream_account.id)

            return StreamingResponse(
                stream_bytes,
                media_type="text/event-stream",
                headers=_stream_headers(stream_account, stream_quota),
            )

        excluded_account_ids = {account.id}
        current_response = upstream_response
        current_attempt = 1
        current_attempt_started_at = attempt_started_at
        current_retry_reason: str | None = None

        while True:
            try:
                response_payload = await asyncio.to_thread(
                    decode_non_stream_response,
                    current_response,
                    model,
                )
            except UpstreamTurnError as exc:
                extra_fields = {"errorCode": exc.error_code or None}
                if current_retry_reason:
                    extra_fields["retryReason"] = current_retry_reason
                _record_attempt(
                    account,
                    quota,
                    request_id,
                    attempt=current_attempt,
                    stream=False,
                    success=False,
                    stop_reason="upstream_turn_error",
                    message=_upstream_turn_error_message(exc),
                    status_code=502,
                    duration_ms=int((time.perf_counter() - current_attempt_started_at) * 1000),
                    extra_fields=extra_fields,
                )
                if not _should_retry_upstream_turn_error(exc):
                    return _anthropic_error_response(
                        502,
                        _upstream_turn_error_message(exc),
                    )
                if _is_retryable_quota_exhausted_turn_error(exc):
                    _mark_account_quota_exhausted_cooldown(store, account)
                next_retry_reason = "upstream_turn_error"
            else:
                usage = response_payload.get("usage") if isinstance(response_payload, dict) else {}
                if not isinstance(usage, dict):
                    usage = {}
                output_summary = _summarize_non_stream_payload(response_payload)
                if not output_summary["empty_response"]:
                    break
                if current_attempt != 1:
                    if disable_on_empty_response:
                        _disable_account_model_on_empty_response(
                            store,
                            account,
                            model,
                        )
                    break
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
                    cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
                    cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
                    success=True,
                    stop_reason=str(response_payload.get("stop_reason") or "end_turn"),
                )
                _record_attempt(
                    account,
                    quota,
                    request_id,
                    attempt=current_attempt,
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
                    duration_ms=int((time.perf_counter() - current_attempt_started_at) * 1000),
                    level="warn",
                    extra_fields={
                        "cacheCreationInputTokens": int(
                            usage.get("cache_creation_input_tokens") or 0
                        ),
                        "cacheReadInputTokens": int(
                            usage.get("cache_read_input_tokens") or 0
                        ),
                        "textChars": int(output_summary["text_chars"] or 0),
                        "toolUseBlocks": int(output_summary["tool_use_blocks"] or 0),
                    },
                )
                api_log_store.record(
                    {
                        "level": "warn",
                        "event": "v1_messages",
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
                        "cacheCreationInputTokens": int(usage.get("cache_creation_input_tokens") or 0),
                        "cacheReadInputTokens": int(usage.get("cache_read_input_tokens") or 0),
                        "remainingQuota": quota.get("remaining_value"),
                        "usedQuota": quota.get("used_value"),
                        "messagesCount": messages_count,
                        "maxTokens": max_tokens,
                        "textChars": int(output_summary["text_chars"] or 0),
                        "toolUseBlocks": int(output_summary["tool_use_blocks"] or 0),
                        "durationMs": int((time.perf_counter() - started_at) * 1000),
                    }
                )
                next_retry_reason = "empty_response"

            try:
                retry_account, retry_quota = await asyncio.to_thread(
                    _select_proxy_account,
                    application,
                    panel_settings,
                    model,
                    exclude_account_ids=excluded_account_ids,
                )
            except ProxySelectionError as exc:
                return _anthropic_error_response(exc.status_code, exc.message)

            retry_body = build_accio_request(
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
                attempt=current_attempt + 1,
                stream=False,
                started_at=retry_started_at,
                record_attempt=_record_attempt,
                build_error_response=_build_upstream_error_response(
                    retry_account,
                    retry_quota,
                    retry_request_id,
                    stream=False,
                ),
                retry_reason=next_retry_reason,
            )
            if isinstance(retry_response, Response):
                return retry_response

            account = retry_account
            quota = retry_quota
            request_id = retry_request_id
            excluded_account_ids.add(account.id)
            current_response = retry_response
            current_attempt += 1
            current_attempt_started_at = retry_started_at
            current_retry_reason = next_retry_reason

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
                else "Anthropic 上游请求完成"
            ),
            status_code=200,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            empty_response=bool(output_summary["empty_response"]),
            duration_ms=int((time.perf_counter() - current_attempt_started_at) * 1000),
            level="warn" if output_summary["empty_response"] else None,
            extra_fields={
                "cacheCreationInputTokens": int(
                    usage.get("cache_creation_input_tokens") or 0
                ),
                "cacheReadInputTokens": int(
                    usage.get("cache_read_input_tokens") or 0
                ),
                "textChars": int(output_summary["text_chars"] or 0),
                "toolUseBlocks": int(output_summary["tool_use_blocks"] or 0),
                "retryReason": None if current_attempt == 1 else current_retry_reason,
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
        usage_stats_store.record_message(
            account_id=account.id,
            model=model,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
            success=True,
            stop_reason=str(response_payload.get("stop_reason") or "end_turn"),
        )
        api_log_store.record(
            {
                "level": "warn" if output_summary["empty_response"] else "info",
                "event": "v1_messages",
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
                    else "非流式调用完成"
                ),
                "statusCode": 200,
                "stopReason": str(response_payload.get("stop_reason") or "end_turn"),
                "inputTokens": int(usage.get("input_tokens") or 0),
                "outputTokens": int(usage.get("output_tokens") or 0),
                "cacheCreationInputTokens": int(usage.get("cache_creation_input_tokens") or 0),
                "cacheReadInputTokens": int(usage.get("cache_read_input_tokens") or 0),
                "remainingQuota": quota.get("remaining_value"),
                "usedQuota": quota.get("used_value"),
                "messagesCount": messages_count,
                "maxTokens": max_tokens,
                "textChars": int(output_summary["text_chars"] or 0),
                "toolUseBlocks": int(output_summary["tool_use_blocks"] or 0),
                "durationMs": int((time.perf_counter() - started_at) * 1000),
            }
        )
        return JSONResponse(response_payload, headers=response_headers)
