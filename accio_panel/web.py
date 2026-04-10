from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import parse_qsl, urlencode, urlparse

import requests
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .api_logs import ApiLogStore
from .app_settings import (
    PanelSettings,
    PanelSettingsStore,
    normalize_api_account_strategy,
    normalize_upstream_proxy_url,
)
from .anthropic_proxy import (
    DEFAULT_ANTHROPIC_MODEL,
    UpstreamTurnError,
    anthropic_error_payload,
    build_accio_request,
    build_models_payload,
    decode_non_stream_response,
    iter_anthropic_sse_bytes,
)
from .client import AccioClient
from .config import Settings
from .gemini_proxy import (
    build_generate_content_request,
    build_gemini_model_payload,
    build_gemini_models_payload,
    decode_gemini_generate_content_response,
    extract_gemini_finish_reason,
    extract_gemini_usage,
    gemini_error_payload,
    iter_gemini_generate_content_sse_bytes,
    normalize_gemini_model_name,
    summarize_gemini_response,
)
from .model_catalog import (
    build_gemini_model_payload_from_catalog,
    build_gemini_models_payload_from_catalog,
    build_openai_models_payload_from_catalog,
    extract_model_catalog,
    is_image_generation_model,
    list_model_names,
    list_proxy_model_names,
)
from .models import Account
from .openai_proxy import (
    build_accio_request_from_openai,
    build_openai_chat_completion_response,
    build_openai_chat_payload_from_responses,
    build_openai_responses_response,
    iter_openai_chat_sse_bytes,
    iter_openai_responses_sse_bytes,
    openai_error_payload,
)
from .persistence import create_runtime_stores
from .store import AccountStore
from .upstream_support import (
    anthropic_stream_chunk_has_meaningful_output as _anthropic_stream_chunk_has_meaningful_output,
    extract_upstream_turn_error_from_chunk as _extract_upstream_turn_error_from_chunk,
    gemini_stream_chunk_has_meaningful_output as _gemini_stream_chunk_has_meaningful_output,
    is_stream_summary_empty as _is_stream_summary_empty,
    make_upstream_attempt_logger as _make_upstream_attempt_logger,
    native_sse_chunk_has_meaningful_output as _native_sse_chunk_has_meaningful_output,
    openai_chat_chunk_has_meaningful_output as _openai_chat_chunk_has_meaningful_output,
    openai_responses_chunk_has_meaningful_output as _openai_responses_chunk_has_meaningful_output,
    prefetch_stream_until_meaningful as _prefetch_stream_until_meaningful,
    record_proxy_log,
    request_upstream_or_error,
    should_retry_upstream_turn_error as _should_retry_upstream_turn_error,
    summarize_non_stream_payload as _summarize_non_stream_payload,
    upstream_turn_error_message as _upstream_turn_error_message,
)
from .usage_stats import UsageStatsStore
from .utils import format_timestamp, mask_token


TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

ENABLED_ACCOUNT_CHECK_INTERVAL_SECONDS = 15 * 60
FAILED_ACCOUNT_RETRY_SECONDS = 5 * 60
RECOVERY_CHECK_BUFFER_SECONDS = 90
ABNORMAL_ACCOUNT_RECOVERY_INTERVAL_SECONDS = 30 * 60
SCHEDULER_TICK_SECONDS = 30
MODEL_CATALOG_CACHE_SECONDS = 60
PAGE_SIZE_OPTIONS = (10, 20, 50)
DEFAULT_PAGE_SIZE = PAGE_SIZE_OPTIONS[0]


class ProxySelectionError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _normalize_success_message(message: Any) -> str:
    if message in (None, "", 0, "0", "success", "ok", "OK"):
        return ""
    return str(message).strip()


def _local_base_url(settings: Settings) -> str:
    return f"http://{settings.callback_host}:{settings.callback_port}"


def _request_base_url(request: Request, settings: Settings) -> str:
    forwarded_proto = str(request.headers.get("x-forwarded-proto") or "").strip()
    forwarded_host = str(request.headers.get("x-forwarded-host") or "").strip()
    host = forwarded_host or str(request.headers.get("host") or "").strip()
    if forwarded_proto and host:
        return f"{forwarded_proto}://{host}"
    base_url = str(request.base_url).rstrip("/")
    return base_url or _local_base_url(settings)


def _is_admin_authenticated(request: Request) -> bool:
    return bool(request.session.get("admin_authenticated"))


def _effective_callback_url(settings: Settings, panel_settings: PanelSettings) -> str:
    return settings.callback_url


def _effective_api_base_url(settings: Settings, panel_settings: PanelSettings) -> str:
    return _local_base_url(settings)


def _parse_callback_payload(callback_value: str) -> dict[str, str]:
    raw_value = str(callback_value or "").strip()
    if not raw_value:
        raise ValueError("请输入完整的回调地址。")

    query = raw_value
    if "?" in raw_value:
        query = raw_value.split("?", 1)[1]
    elif raw_value.startswith("http://") or raw_value.startswith("https://"):
        parsed = urlparse(raw_value)
        query = parsed.query

    if not query:
        raise ValueError("回调地址中缺少查询参数。")

    params = {
        key: value
        for key, value in parse_qsl(query.lstrip("?"), keep_blank_values=True)
    }
    if not params.get("accessToken") or not params.get("refreshToken"):
        raise ValueError("回调地址缺少 accessToken 或 refreshToken。")
    return params


def _activate_callback_account(
    client: AccioClient,
    account: Account,
    panel_settings: PanelSettings,
) -> dict[str, Any]:
    return client.activate_account(
        account,
        proxy_url=panel_settings.upstream_proxy_url,
    )


def _activation_summary_text(activation: dict[str, Any]) -> str:
    message = str(activation.get("message") or "").strip()
    if message:
        return message
    return "账号激活完成" if activation.get("success") else "账号激活未完成"


def _import_callback_account(
    store: AccountStore,
    client: AccioClient,
    panel_settings: PanelSettings,
    *,
    access_token: str,
    refresh_token: str,
    expires_at: str | int | None,
    cookie: str | None,
) -> tuple[Account, dict[str, Any], bool, dict[str, Any]]:
    account, created = store.upsert_from_callback(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        cookie=cookie,
    )
    activation = _activate_callback_account(client, account, panel_settings)
    account, quota = _query_quota_with_refresh_fallback(
        store,
        client,
        account,
        panel_settings,
    )
    return account, quota, created, activation


def _now_timestamp() -> int:
    return int(time.time())


def _normalize_target_model(
    model_name: str | None,
    *,
    provider: str | None = None,
) -> str:
    if provider == "gemini":
        return normalize_gemini_model_name(model_name)
    return str(model_name or "").strip().lower()


def _disabled_model_items(account: Account) -> list[dict[str, str]]:
    disabled_models = (
        account.disabled_models if isinstance(account.disabled_models, dict) else {}
    )
    return [
        {"model": model_name, "reason": str(reason or "").strip()}
        for model_name, reason in sorted(disabled_models.items())
        if str(model_name or "").strip()
    ]


def _account_model_disabled_reason(
    account: Account,
    model_name: str | None,
    *,
    provider: str | None = None,
) -> str | None:
    normalized_model = _normalize_target_model(model_name, provider=provider)
    if not normalized_model:
        return None

    disabled_models = (
        account.disabled_models if isinstance(account.disabled_models, dict) else {}
    )
    reason = str(disabled_models.get(normalized_model) or "").strip()
    return reason or None


def _disable_account_model_on_empty_response(
    store: AccountStore,
    account: Account,
    model_name: str,
    *,
    provider: str | None = None,
) -> Account:
    normalized_model = _normalize_target_model(model_name, provider=provider)
    if not normalized_model:
        return account

    if _account_model_disabled_reason(account, normalized_model, provider=provider):
        return account

    reason = f"模型 {normalized_model} 出现空回复，已自动禁用该账号调用此模型。"
    updated = store.set_disabled_model(account.id, normalized_model, reason)
    return updated or account


def _api_account_strategy_label(strategy: str) -> str:
    return "轮询" if strategy == "round_robin" else "优先填充"


def _initial_model_catalog_cache() -> dict[str, Any]:
    return {
        "entries": [],
        "expiresAt": 0.0,
        "loadedAt": 0.0,
        "sourceAccountId": "",
        "error": "",
    }


def _sorted_enabled_accounts(store: AccountStore) -> list[Account]:
    return sorted(
        _ordered_proxy_candidates(store),
        key=lambda item: (item.fill_priority, item.name, item.id),
    )


def _should_disable_model_on_empty_response(
    payload: Any,
    model_name: str,
) -> bool:
    if str(model_name or "").strip().lower().startswith("claude"):
        return False
    return True


def _empty_response_log_message(
    model_name: str,
    *,
    disable_model: bool,
) -> str:
    if disable_model:
        return f"空回复，已禁用模型 {model_name}"
    return "空回复"


def _query_llm_config_with_refresh_fallback(
    store: AccountStore,
    client: AccioClient,
    account: Account,
    panel_settings: PanelSettings,
) -> tuple[Account, dict[str, Any]]:
    config_result = client.query_llm_config(
        account,
        proxy_url=panel_settings.upstream_proxy_url,
    )
    if config_result.get("success"):
        return account, config_result

    refresh_result = _refresh_token(client, account, panel_settings)
    if not refresh_result.get("success"):
        return account, config_result

    refreshed_data = refresh_result.get("data") or {}
    updated_account = store.update_tokens(
        account.id,
        access_token=str(refreshed_data.get("accessToken") or account.access_token),
        refresh_token=str(refreshed_data.get("refreshToken") or account.refresh_token),
        expires_at=refreshed_data.get("expiresAt"),
    )
    if updated_account is None:
        return account, config_result
    account = updated_account
    retried_result = client.query_llm_config(
        account,
        proxy_url=panel_settings.upstream_proxy_url,
    )
    return account, retried_result


def _load_dynamic_model_catalog(
    application: FastAPI,
    panel_settings: PanelSettings,
) -> tuple[list[dict[str, Any]], str]:
    cache = getattr(application.state, "model_catalog_cache", None)
    if not isinstance(cache, dict):
        cache = _initial_model_catalog_cache()
        application.state.model_catalog_cache = cache

    now_value = time.time()
    cached_entries = cache.get("entries")
    if isinstance(cached_entries, list) and cached_entries and float(cache.get("expiresAt") or 0) > now_value:
        return list(cached_entries), "cache"

    store: AccountStore = application.state.store
    client: AccioClient = application.state.client
    errors: list[str] = []
    for candidate in _sorted_enabled_accounts(store):
        account, config_result = _query_llm_config_with_refresh_fallback(
            store,
            client,
            candidate,
            panel_settings,
        )
        entries = extract_model_catalog(config_result) if isinstance(config_result, dict) else []
        if entries:
            cache.update(
                {
                    "entries": list(entries),
                    "expiresAt": now_value + MODEL_CATALOG_CACHE_SECONDS,
                    "loadedAt": now_value,
                    "sourceAccountId": account.id,
                    "error": "",
                }
            )
            return list(entries), "live"
        if isinstance(config_result, dict):
            message = str(config_result.get("message") or "模型配置为空")
            errors.append(f"{account.name}: {message}")

    if isinstance(cached_entries, list) and cached_entries:
        cache["error"] = errors[0] if errors else "模型目录刷新失败，已回退到旧缓存。"
        return list(cached_entries), "stale"

    cache["error"] = errors[0] if errors else "当前没有可用账号可拉取模型目录。"
    return [], "unavailable"


def _dynamic_proxy_model_names(
    application: FastAPI,
    panel_settings: PanelSettings,
) -> set[str]:
    entries, _ = _load_dynamic_model_catalog(application, panel_settings)
    return list_proxy_model_names(entries)


def _dynamic_gemini_model_names(
    application: FastAPI,
    panel_settings: PanelSettings,
) -> set[str]:
    entries, _ = _load_dynamic_model_catalog(application, panel_settings)
    return list_model_names(entries, provider="gemini")


def _resolve_gemini_model_payload(
    application: FastAPI,
    panel_settings: PanelSettings,
    model_name: str,
) -> tuple[dict[str, Any] | None, str]:
    normalized_name = normalize_gemini_model_name(model_name)
    catalog, source = _load_dynamic_model_catalog(application, panel_settings)
    if catalog:
        payload = build_gemini_model_payload_from_catalog(catalog, normalized_name)
        if payload:
            return payload, source
    return build_gemini_model_payload(normalized_name), "static-fallback"


def _model_catalog_dashboard_text(
    entries: list[dict[str, Any]],
    source: str,
) -> str:
    if not entries:
        return "动态模型目录暂不可用，列表接口会回退到内置静态模型。"

    names = sorted(list_model_names(entries))
    preview = names[:8]
    preview_text = " / ".join(preview)
    if len(names) > len(preview):
        preview_text = f"{preview_text} / ...（共 {len(names)} 个）"

    source_label = {
        "live": "实时",
        "cache": "缓存",
        "stale": "旧缓存",
        "unavailable": "不可用",
    }.get(source, source)
    return f"{preview_text}（{source_label}）"


def _is_allowed_dynamic_model(
    application: FastAPI,
    panel_settings: PanelSettings,
    model_name: str,
    *,
    provider: str | None = None,
) -> tuple[bool, list[str]]:
    normalized = (
        normalize_gemini_model_name(model_name)
        if provider == "gemini"
        else str(model_name or "").strip()
    )
    if not normalized:
        return False, []
    available = (
        sorted(_dynamic_gemini_model_names(application, panel_settings))
        if provider == "gemini"
        else sorted(_dynamic_proxy_model_names(application, panel_settings))
    )
    if provider != "gemini" and is_image_generation_model(normalized):
        return False, available
    if available:
        return normalized in set(available), available
    return True, []


def _proxy_fill_sort_key(account: Account, quota: dict[str, Any]) -> tuple[Any, ...]:
    return (
        account.fill_priority,
        quota["remaining_value"],
        account.name,
        account.id,
    )


def _account_status_view(account: Account) -> dict[str, Any]:
    if not account.manual_enabled:
        disabled_reason = str(account.auto_disabled_reason or "").strip()
        if disabled_reason:
            return {
                "effective_enabled": False,
                "label": "异常禁用",
                "level": "auto",
                "hint": disabled_reason,
            }
        return {
            "effective_enabled": False,
            "label": "手动禁用",
            "level": "manual",
            "hint": "该账号已被手动禁用，不参与启用状态。",
        }

    if account.auto_disabled:
        return {
            "effective_enabled": False,
            "label": "自动禁用",
            "level": "auto",
            "hint": account.auto_disabled_reason or "额度耗尽后已自动禁用。",
        }

    return {
        "effective_enabled": True,
        "label": "已启用",
        "level": "enabled",
        "hint": "账号当前处于启用状态。",
    }


def _unauthorized_json() -> JSONResponse:
    return JSONResponse(
        {"success": False, "message": "请先输入管理员密码登录"},
        status_code=401,
    )


def _anthropic_error_response(
    status_code: int,
    message: str,
    *,
    error_type: str = "api_error",
) -> JSONResponse:
    return JSONResponse(
        anthropic_error_payload(message, error_type=error_type),
        status_code=status_code,
    )


def _gemini_error_response(
    status_code: int,
    message: str,
    *,
    error_status: str = "INVALID_ARGUMENT",
) -> JSONResponse:
    return JSONResponse(
        gemini_error_payload(status_code, message, error_status=error_status),
        status_code=status_code,
    )


def _openai_error_response(
    status_code: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        openai_error_payload(message, error_type=error_type, code=code),
        status_code=status_code,
    )


def _native_error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"success": False, "message": message},
        status_code=status_code,
    )


def _extract_proxy_api_key(request: Request) -> str:
    api_key = str(request.headers.get("x-api-key") or "").strip()
    if api_key:
        return api_key

    api_key = str(request.headers.get("x-goog-api-key") or "").strip()
    if api_key:
        return api_key

    authorization = str(request.headers.get("authorization") or "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()

    from urllib.parse import unquote

    raw_query = str(request.url.query or "")
    for pair in raw_query.split("&"):
        if not pair:
            continue
        name, separator, value = pair.partition("=")
        if not separator:
            continue
        normalized_name = unquote(name).strip().lower()
        if normalized_name in {"key", "api_key", "x-api-key"}:
            return unquote(value).strip()
    return ""


def _authorize_proxy_request(
    request: Request,
    panel_settings: PanelSettings,
) -> JSONResponse | None:
    api_key = _extract_proxy_api_key(request)
    if api_key == panel_settings.admin_password:
        return None
    return _anthropic_error_response(
        401,
        "无效的 API Key，请使用管理员密码作为 x-api-key 或 Bearer Token。",
        error_type="authentication_error",
    )


def _iter_upstream_sse_bytes(response: requests.Response) -> Iterator[bytes]:
    try:
        for raw_line in response.iter_lines(decode_unicode=False):
            if raw_line is None:
                continue
            line = raw_line.encode("utf-8") if isinstance(raw_line, str) else raw_line
            if not line:
                continue
            yield line + b"\n\n"
    finally:
        response.close()


def _ordered_proxy_candidates(
    store: AccountStore,
    model_name: str | None = None,
    *,
    provider: str | None = None,
    exclude_account_ids: set[str] | None = None,
) -> list[Account]:
    excluded_ids = exclude_account_ids or set()
    return [
        account
        for account in store.list_accounts()
        if account.manual_enabled
        and not account.auto_disabled
        and account.id not in excluded_ids
        and not _account_model_disabled_reason(
            account,
            model_name,
            provider=provider,
        )
    ]


def _query_quota(
    client: AccioClient,
    account: Account,
    panel_settings: PanelSettings,
) -> dict[str, Any]:
    return client.query_quota(
        account,
        proxy_url=panel_settings.upstream_proxy_url,
    )


def _refresh_token(
    client: AccioClient,
    account: Account,
    panel_settings: PanelSettings,
) -> dict[str, Any]:
    return client.refresh_token(
        account,
        proxy_url=panel_settings.upstream_proxy_url,
    )


def _disable_account_after_refresh_failure(
    store: AccountStore,
    account: Account,
    reason: str,
) -> Account:
    now_ts = _now_timestamp()
    updated = store.set_manual_enabled(account.id, False)
    if not updated:
        account.manual_enabled = False
        account.auto_disabled = False
        account.auto_disabled_reason = reason
        account.last_quota_check_at = now_ts
        account.next_quota_check_at = now_ts + ABNORMAL_ACCOUNT_RECOVERY_INTERVAL_SECONDS
        account.next_quota_check_reason = "异常禁用后定时恢复检查"
        return account
    updated.auto_disabled = False
    updated.auto_disabled_reason = reason
    updated.last_quota_check_at = now_ts
    updated.next_quota_check_at = now_ts + ABNORMAL_ACCOUNT_RECOVERY_INTERVAL_SECONDS
    updated.next_quota_check_reason = "异常禁用后定时恢复检查"
    store.save(updated)
    return updated


def _query_quota_with_refresh_fallback(
    store: AccountStore,
    client: AccioClient,
    account: Account,
    panel_settings: PanelSettings,
) -> tuple[Account, dict[str, Any]]:
    quota_result = _query_quota(client, account, panel_settings)
    if not account.manual_enabled:
        return _apply_quota_result(store, account, quota_result, panel_settings)
    if quota_result.get("success"):
        return _apply_quota_result(store, account, quota_result, panel_settings)

    refresh_result = _refresh_token(client, account, panel_settings)
    if not refresh_result.get("success"):
        reason = (
            "额度查询失败，且 Token 刷新失败："
            f"{refresh_result.get('message') or '刷新失败'}。系统已自动禁用该账号，请手动处理。"
        )
        disabled_account = _disable_account_after_refresh_failure(store, account, reason)
        failed_quota = _build_quota_view(quota_result)
        failed_quota["message"] = reason
        return disabled_account, failed_quota

    refreshed_data = refresh_result.get("data") or {}
    updated_account = store.update_tokens(
        account.id,
        access_token=str(refreshed_data.get("accessToken") or account.access_token),
        refresh_token=str(refreshed_data.get("refreshToken") or account.refresh_token),
        expires_at=refreshed_data.get("expiresAt"),
    )
    if updated_account:
        updated_account.next_quota_check_at = _now_timestamp()
        updated_account.next_quota_check_reason = "额度查询失败后自动刷新 Token 并重试额度"
        store.save(updated_account)
        account = updated_account

    retried_quota_result = _query_quota(client, account, panel_settings)
    account, quota = _apply_quota_result(store, account, retried_quota_result, panel_settings)
    if quota["success"]:
        quota["message"] = quota["message"] or "额度查询失败后已自动刷新 Token 并恢复。"
    else:
        retry_message = str(quota.get("message") or "").strip()
        quota["message"] = (
            "额度查询失败，已自动刷新 Token 并重试，但额度查询仍失败。"
            + (f" {retry_message}" if retry_message else "")
        ).strip()
    return account, quota


def _try_recover_abnormal_account(
    store: AccountStore,
    client: AccioClient,
    account: Account,
    panel_settings: PanelSettings,
) -> tuple[Account, dict[str, Any]]:
    """尝试恢复异常禁用的账号：先刷新 Token，再查询额度。"""
    now_ts = _now_timestamp()

    refresh_result = _refresh_token(client, account, panel_settings)
    if not refresh_result.get("success"):
        account.last_quota_check_at = now_ts
        account.next_quota_check_at = now_ts + ABNORMAL_ACCOUNT_RECOVERY_INTERVAL_SECONDS
        account.next_quota_check_reason = "异常禁用恢复检查失败，等待下次重试"
        store.save(account)
        return account, {"success": False, "message": "Token 刷新仍然失败"}

    refreshed_data = refresh_result.get("data") or {}
    updated_account = store.update_tokens(
        account.id,
        access_token=str(refreshed_data.get("accessToken") or account.access_token),
        refresh_token=str(refreshed_data.get("refreshToken") or account.refresh_token),
        expires_at=refreshed_data.get("expiresAt"),
    )
    if updated_account:
        account = updated_account

    account.manual_enabled = True
    account.auto_disabled = False
    account.auto_disabled_reason = None
    store.save(account)

    quota_result = _query_quota(client, account, panel_settings)
    return _apply_quota_result(store, account, quota_result, panel_settings)


def _check_proxy_candidate(
    store: AccountStore,
    client: AccioClient,
    panel_settings: PanelSettings,
    account: Account,
) -> tuple[Account, dict[str, Any]]:
    return _query_quota_with_refresh_fallback(
        store,
        client,
        account,
        panel_settings,
    )


def _cached_quota_view(account: Account) -> dict[str, Any]:
    remaining = account.last_remaining_quota
    if remaining is not None and remaining >= 0:
        total = account.last_total_quota or 0
        used = max(0, total - remaining) if total > 0 else 0
        remaining_ratio = (
            max(0, min(100, round((remaining / total) * 100)))
            if total > 0
            else 0
        )
        level = "low"
        if remaining_ratio < 20:
            level = "high"
        elif remaining_ratio < 50:
            level = "medium"
        return {
            "success": True,
            "used_value": used,
            "used_text": f"{used}/{total}" if total > 0 else "-",
            "remaining_value": remaining,
            "remaining_ratio": remaining_ratio,
            "remaining_text": f"{remaining}/{total}" if total > 0 else str(remaining),
            "reset_text": "-",
            "level": level,
            "message": "",
        }
    return {
        "success": False,
        "used_value": 0,
        "used_text": "-",
        "remaining_value": 0,
        "remaining_ratio": 0,
        "remaining_text": "缓存不可用",
        "reset_text": "-",
        "level": "error",
        "message": "",
    }


def _select_proxy_account(
    application: FastAPI,
    panel_settings: PanelSettings,
    model_name: str | None = None,
    *,
    provider: str | None = None,
    exclude_account_ids: set[str] | None = None,
) -> tuple[Account, dict[str, Any]]:
    store: AccountStore = application.state.store

    candidates = _ordered_proxy_candidates(
        store,
        model_name,
        provider=provider,
        exclude_account_ids=exclude_account_ids,
    )
    if not candidates:
        if model_name:
            raise ProxySelectionError(
                503,
                f"当前没有已启用账号可用于模型 {model_name}。",
            )
        raise ProxySelectionError(503, "当前没有已启用的账号可供 API 调用。")

    errors: list[str] = []
    strategy = panel_settings.api_account_strategy

    if strategy == "round_robin":
        start_index = application.state.proxy_round_robin_index % len(candidates)
        for offset in range(len(candidates)):
            index = (start_index + offset) % len(candidates)
            account = candidates[index]
            if account.auto_disabled:
                errors.append(
                    f"{account.name}: {account.auto_disabled_reason or '账号已自动禁用。'}"
                )
                continue
            remaining = account.last_remaining_quota
            if remaining is not None and remaining <= 0:
                errors.append(f"{account.name}: 剩余额度为 0%")
                continue
            application.state.proxy_round_robin_index = (index + 1) % len(candidates)
            return account, _cached_quota_view(account)

        application.state.proxy_round_robin_index = 0
    else:
        available = [
            a for a in candidates
            if not a.auto_disabled
            and (a.last_remaining_quota is None or a.last_remaining_quota > 0)
        ]
        if available:
            available.sort(
                key=lambda a: (
                    a.fill_priority,
                    -(a.last_remaining_quota or 0),
                    a.name,
                    a.id,
                )
            )
            return available[0], _cached_quota_view(available[0])

        for account in sorted(
            candidates,
            key=lambda a: (a.fill_priority, a.name, a.id),
        ):
            if account.auto_disabled:
                errors.append(
                    f"{account.name}: {account.auto_disabled_reason or '账号已自动禁用。'}"
                )
            elif account.last_remaining_quota is not None and account.last_remaining_quota <= 0:
                errors.append(f"{account.name}: 剩余额度为 0%")

    raise ProxySelectionError(
        503,
        errors[0] if errors else "当前没有可用账号可供 API 调用。",
    )


def _parse_dashboard_view(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "settings":
        return "settings"
    if normalized == "stats":
        return "stats"
    if normalized == "logs":
        return "logs"
    return "accounts"


def _parse_page_size(value: str | None) -> int:
    try:
        page_size = int(str(value or DEFAULT_PAGE_SIZE))
    except (TypeError, ValueError):
        return DEFAULT_PAGE_SIZE
    return page_size if page_size in PAGE_SIZE_OPTIONS else DEFAULT_PAGE_SIZE


def _parse_page_number(value: str | None) -> int:
    try:
        page_number = int(str(value or "1"))
    except (TypeError, ValueError):
        return 1
    return max(1, page_number)


def _build_page_numbers(current_page: int, total_pages: int) -> list[int]:
    if total_pages <= 7:
        return list(range(1, total_pages + 1))
    start = max(1, current_page - 2)
    end = min(total_pages, start + 4)
    start = max(1, end - 4)
    return list(range(start, end + 1))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _extract_subscription_entitlement(data: dict[str, Any]) -> dict[str, Any]:
    entitlement = data.get("entitlement")
    if not isinstance(entitlement, dict):
        return {}

    for key in ("monthly", "referral", "daily"):
        item = entitlement.get(key)
        if isinstance(item, dict) and any(
            item.get(field) not in (None, "")
            for field in ("total", "used", "remaining", "nextBillingDate")
        ):
            return item
    return {}


def _parse_billing_timestamp(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None

    normalized = text.replace("T", " ").replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return int(datetime.strptime(normalized, fmt).timestamp())
        except ValueError:
            continue
    return None


def _build_quota_view(result: dict[str, Any]) -> dict[str, Any]:
    data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(data, dict):
        data = {}

    entitlement = _extract_subscription_entitlement(data)
    total_value = max(
        0,
        _as_int(data.get("total"), _as_int(entitlement.get("total"))),
    )
    remaining_value = max(
        0,
        _as_int(data.get("remaining"), _as_int(entitlement.get("remaining"))),
    )
    used_value = max(
        0,
        _as_int(
            entitlement.get("used"),
            max(0, total_value - remaining_value),
        ),
    )
    if total_value <= 0 and (used_value > 0 or remaining_value > 0):
        total_value = used_value + remaining_value

    remaining_ratio = (
        max(0, min(100, round((remaining_value / total_value) * 100)))
        if total_value > 0
        else 0
    )
    next_billing_text = str(entitlement.get("nextBillingDate") or "").strip()

    if result.get("success"):
        level = "low"
        if remaining_ratio < 20:
            level = "high"
        elif remaining_ratio < 50:
            level = "medium"
        return {
            "success": True,
            "total_value": total_value,
            "used_value": used_value,
            "used_text": (
                f"{used_value}/{total_value}"
                if total_value > 0
                else str(used_value)
            ),
            "remaining_value": remaining_value,
            "remaining_ratio": remaining_ratio,
            "remaining_text": (
                f"{remaining_value}/{total_value}"
                if total_value > 0
                else str(remaining_value)
            ),
            "reset_text": next_billing_text or "-",
            "level": level,
            "message": _normalize_success_message(result.get("message")),
        }

    return {
        "success": False,
        "total_value": 0,
        "used_value": 0,
        "used_text": "-",
        "remaining_value": 0,
        "remaining_ratio": 0,
        "remaining_text": "获取失败",
        "reset_text": "-",
        "level": "error",
        "message": result.get("message") or "额度查询失败",
    }


def _extract_next_billing_timestamp(result: dict[str, Any]) -> int | None:
    data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(data, dict):
        return None

    entitlement = _extract_subscription_entitlement(data)
    return _parse_billing_timestamp(entitlement.get("nextBillingDate"))


def _plan_next_quota_check(
    account: Account,
    *,
    quota_success: bool,
    next_billing_at: int | None,
    panel_settings: PanelSettings,
    now_ts: int,
) -> tuple[int | None, str | None]:
    if not account.manual_enabled:
        return None, None

    if not quota_success:
        return now_ts + FAILED_ACCOUNT_RETRY_SECONDS, "额度查询失败后重试"

    if account.auto_disabled:
        if not panel_settings.auto_enable_on_recovered_quota:
            return None, None
        if next_billing_at is not None:
            return (
                max(now_ts + RECOVERY_CHECK_BUFFER_SECONDS, next_billing_at + RECOVERY_CHECK_BUFFER_SECONDS),
                "等待额度重置后自动恢复检查",
            )
        return now_ts + FAILED_ACCOUNT_RETRY_SECONDS, "自动恢复重试"

    return now_ts + ENABLED_ACCOUNT_CHECK_INTERVAL_SECONDS, "启用账号额度巡检"


def _apply_quota_result(
    store: AccountStore,
    account: Account,
    quota_result: dict[str, Any],
    panel_settings: PanelSettings,
) -> tuple[Account, dict[str, Any]]:
    quota = _build_quota_view(quota_result)
    next_billing_at = _extract_next_billing_timestamp(quota_result)
    now_ts = _now_timestamp()
    should_mark_updated = False

    if quota["success"]:
        account.last_remaining_quota = quota["remaining_value"]
        account.last_total_quota = quota.get("total_value")
        if (
            panel_settings.auto_disable_on_empty_quota
            and account.manual_enabled
            and quota["remaining_value"] <= 0
            and not account.auto_disabled
        ):
            account.auto_disabled = True
            account.auto_disabled_reason = "剩余额度已耗尽，系统已自动禁用。"
            should_mark_updated = True
        elif (
            panel_settings.auto_enable_on_recovered_quota
            and account.auto_disabled
            and quota["remaining_value"] > 0
        ):
            account.auto_disabled = False
            account.auto_disabled_reason = None
            should_mark_updated = True

    account.last_quota_check_at = now_ts
    (
        account.next_quota_check_at,
        account.next_quota_check_reason,
    ) = _plan_next_quota_check(
        account,
        quota_success=quota["success"],
        next_billing_at=next_billing_at,
        panel_settings=panel_settings,
        now_ts=now_ts,
    )

    if should_mark_updated:
        account.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))

    store.save(account)
    return account, quota


def _build_dashboard_items(
    accounts: list[Account],
    client: AccioClient,
    store: AccountStore,
    panel_settings: PanelSettings,
) -> list[dict[str, Any]]:
    if not accounts:
        return []

    quota_map: dict[str, tuple[Account, dict[str, Any]]] = {}
    workers = min(8, len(accounts))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _query_quota_with_refresh_fallback,
                store,
                client,
                account,
                panel_settings,
            ): account.id
            for account in accounts
        }
        for future in as_completed(future_map):
            account_id = future_map[future]
            try:
                quota_map[account_id] = future.result()
            except Exception as exc:  # pragma: no cover
                fallback_account = store.get_account(account_id)
                if fallback_account is None:
                    fallback_account = next(
                        (candidate for candidate in accounts if candidate.id == account_id),
                        None,
                    )
                if fallback_account is None:
                    continue
                quota_map[account_id] = (
                    fallback_account,
                    _build_quota_view({"success": False, "message": str(exc)}),
                )

    items: list[dict[str, Any]] = []
    for account in accounts:
        account, quota_view = quota_map.get(
            account.id,
            (
                account,
                _build_quota_view({"success": False, "message": "额度查询失败"}),
            ),
        )
        status = _account_status_view(account)
        items.append(
            {
                "id": account.id,
                "name": account.name,
                "utdid": account.utdid,
                "fill_priority": account.fill_priority,
                "masked_access_token": mask_token(account.access_token),
                "expires_at_text": format_timestamp(account.expires_at),
                "added_at": account.added_at,
                "updated_at": account.updated_at,
                "manual_enabled": account.manual_enabled,
                "auto_disabled": account.auto_disabled,
                "next_quota_check_at": account.next_quota_check_at,
                "next_quota_check_reason": account.next_quota_check_reason,
                "status": status,
                "quota": quota_view,
            }
        )
    return items


async def _quota_scheduler_loop(application: FastAPI) -> None:
    store: AccountStore = application.state.store
    client: AccioClient = application.state.client
    panel_settings_store: PanelSettingsStore = application.state.panel_settings_store

    while True:
        panel_settings = panel_settings_store.load()
        now_ts = _now_timestamp()
        accounts = store.list_accounts()

        due_accounts: list[Account] = []
        abnormal_recovery_accounts: list[Account] = []

        for account in accounts:
            if not account.manual_enabled:
                # 异常禁用的账号：有 reason 且有调度时间，到期后尝试恢复
                if (
                    account.auto_disabled_reason
                    and account.next_quota_check_at is not None
                    and account.next_quota_check_at <= now_ts
                ):
                    abnormal_recovery_accounts.append(account)
                elif (
                    not account.auto_disabled_reason
                    and (
                        account.next_quota_check_at is not None
                        or account.next_quota_check_reason is not None
                    )
                ):
                    account.next_quota_check_at = None
                    account.next_quota_check_reason = None
                    store.save(account)
                continue

            if account.next_quota_check_at is None or account.next_quota_check_at <= now_ts:
                due_accounts.append(account)

        if due_accounts:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    None,
                    _query_quota_with_refresh_fallback,
                    store,
                    client,
                    account,
                    panel_settings,
                )
                for account in due_accounts
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        for account in abnormal_recovery_accounts:
            await asyncio.to_thread(
                _try_recover_abnormal_account,
                store,
                client,
                account,
                panel_settings,
            )

        await asyncio.sleep(SCHEDULER_TICK_SECONDS)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store, panel_settings_store = create_runtime_stores(settings)
    client = AccioClient(settings)
    usage_stats_store = UsageStatsStore(settings.stats_file)
    api_log_store = ApiLogStore(settings.api_logs_file)
    initial_panel_settings = panel_settings_store.load()

    @contextlib.asynccontextmanager
    async def lifespan(application: FastAPI):
        task = application.state.quota_scheduler_task
        if task is None or task.done():
            application.state.quota_scheduler_task = asyncio.create_task(
                _quota_scheduler_loop(application)
            )
        try:
            yield
        finally:
            task = application.state.quota_scheduler_task
            if task is None:
                return
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            application.state.quota_scheduler_task = None

    application = FastAPI(
        title="Accio 多账号管理面板",
        description="支持 Token 刷新、额度查看、登录链接获取与 FastAPI 回调保存。",
        version=settings.version,
        lifespan=lifespan,
    )
    application.add_middleware(
        SessionMiddleware,
        secret_key=initial_panel_settings.session_secret,
        session_cookie="accio_admin_session",
        same_site="lax",
        max_age=60 * 60 * 24 * 7,
        https_only=False,
    )
    if settings.allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "x-accio-model-source",
                "x-accio-account-id",
                "x-accio-account-strategy",
                "x-accio-account-remaining",
            ],
        )
    application.state.settings = settings
    application.state.store = store
    application.state.client = client
    application.state.usage_stats_store = usage_stats_store
    application.state.api_log_store = api_log_store
    application.state.panel_settings_store = panel_settings_store
    application.state.storage_backend = settings.storage_backend
    application.state.quota_scheduler_task = None
    application.state.proxy_round_robin_index = 0
    application.state.model_catalog_cache = _initial_model_catalog_cache()

    @application.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=307)

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        panel_settings = panel_settings_store.load()
        callback_url = _effective_callback_url(settings, panel_settings)
        if not _is_admin_authenticated(request):
            return TEMPLATES.TemplateResponse(
                request=request,
                name="dashboard_public.html",
                context={
                    "page_title": "Accio 多账号管理面板",
                    "callback_url": callback_url,
                    "oauth_url": "/oauth",
                },
            )

        current_view = _parse_dashboard_view(request.query_params.get("view"))
        page_size = _parse_page_size(request.query_params.get("pageSize"))
        requested_page = _parse_page_number(request.query_params.get("page"))
        all_accounts = store.list_accounts()
        account_count = len(all_accounts)
        enabled_accounts = [
            a for a in all_accounts
            if a.manual_enabled and not a.auto_disabled
        ]
        enabled_account_count = len(enabled_accounts)
        enabled_quota_values = [
            a.last_remaining_quota
            for a in enabled_accounts
            if a.last_remaining_quota is not None
        ]
        total_remaining_quota = (
            sum(enabled_quota_values)
            if enabled_quota_values
            else None
        )
        total_pages = max(1, ((account_count - 1) // page_size) + 1) if account_count else 1
        current_page = min(requested_page, total_pages)
        page_start = (current_page - 1) * page_size
        page_end = page_start + page_size
        page_accounts = all_accounts[page_start:page_end] if current_view == "accounts" else []
        model_catalog, model_catalog_source = _load_dynamic_model_catalog(
            application,
            panel_settings,
        )
        stats_snapshot = usage_stats_store.snapshot(
            {account.id: account.name for account in all_accounts}
        )
        api_logs = api_log_store.recent(200) if current_view == "logs" else []
        dashboard_items = _build_dashboard_items(
            page_accounts,
            client,
            store,
            panel_settings,
        )
        api_base_url = _request_base_url(request, settings)
        return TEMPLATES.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "page_title": "Accio 多账号管理面板",
                "accounts": dashboard_items,
                "account_count": account_count,
                "enabled_account_count": enabled_account_count,
                "total_remaining_quota": total_remaining_quota,
                "callback_url": callback_url,
                "oauth_url": "/oauth",
                "upstream_proxy_url": panel_settings.upstream_proxy_url,
                "version": settings.version,
                "base_url": settings.base_url,
                "api_base_url": api_base_url,
                "admin_password": panel_settings.admin_password,
                "auto_disable_on_empty_quota": panel_settings.auto_disable_on_empty_quota,
                "auto_enable_on_recovered_quota": panel_settings.auto_enable_on_recovered_quota,
                "api_account_strategy": panel_settings.api_account_strategy,
                "api_account_strategy_label": _api_account_strategy_label(
                    panel_settings.api_account_strategy
                ),
                "supported_models_text": _model_catalog_dashboard_text(
                    model_catalog,
                    model_catalog_source,
                ),
                "current_view": current_view,
                "current_page": current_page,
                "page_size": page_size,
                "page_size_options": PAGE_SIZE_OPTIONS,
                "total_pages": total_pages,
                "page_numbers": _build_page_numbers(current_page, total_pages),
                "page_start_index": page_start + 1 if page_accounts else 0,
                "page_end_index": page_start + len(page_accounts),
                "usage_stats": stats_snapshot,
                "api_logs": api_logs,
            },
        )

    @application.get("/settings", include_in_schema=False)
    def settings_page() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=307)

    @application.post("/api/auth/login")
    def admin_login(request: Request, payload: dict[str, Any] = Body(...)) -> JSONResponse:
        panel_settings = panel_settings_store.load()
        password = str(payload.get("password") or "")
        if password != panel_settings.admin_password:
            return JSONResponse(
                {"success": False, "message": "管理员密码错误"},
                status_code=401,
            )

        request.session["admin_authenticated"] = True
        return JSONResponse({"success": True, "message": "登录成功"})

    @application.post("/api/auth/logout")
    def admin_logout(request: Request) -> JSONResponse:
        request.session.clear()
        return JSONResponse({"success": True, "message": "已退出登录"})

    @application.get("/oauth", response_class=HTMLResponse)
    def oauth_page(request: Request) -> HTMLResponse:
        panel_settings = panel_settings_store.load()
        callback_url = _effective_callback_url(settings, panel_settings)
        return TEMPLATES.TemplateResponse(
            request=request,
            name="oauth.html",
            context={
                "page_title": "Accio OAuth 登录",
                "callback_url": callback_url,
                "login_url": client.build_login_url(callback_url),
                "dashboard_url": "/dashboard",
            },
        )

    @application.get("/login")
    def login_redirect() -> RedirectResponse:
        panel_settings = panel_settings_store.load()
        callback_url = _effective_callback_url(settings, panel_settings)
        return RedirectResponse(client.build_login_url(callback_url), status_code=307)

    @application.get("/api/login-link")
    def api_login_link() -> JSONResponse:
        panel_settings = panel_settings_store.load()
        callback_url = _effective_callback_url(settings, panel_settings)
        return JSONResponse(
            {
                "success": True,
                "url": client.build_login_url(callback_url),
                "callbackUrl": callback_url,
            }
        )

    @application.post("/api/oauth/import-callback")
    def import_callback_url(
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        callback_url = str(payload.get("callbackUrl") or "").strip()
        try:
            callback_payload = _parse_callback_payload(callback_url)
        except ValueError as exc:
            return JSONResponse(
                {"success": False, "message": str(exc)},
                status_code=400,
            )

        panel_settings = panel_settings_store.load()
        account, quota, created, activation = _import_callback_account(
            store,
            client,
            panel_settings,
            access_token=callback_payload.get("accessToken", ""),
            refresh_token=callback_payload.get("refreshToken", ""),
            expires_at=callback_payload.get("expiresAt"),
            cookie=callback_payload.get("cookie"),
        )
        return JSONResponse(
            {
                "success": True,
                "message": (
                    ("账号已导入到面板。" if created else "账号已存在，Token 已更新。")
                    + " "
                    + _activation_summary_text(activation)
                ).strip(),
                "account": {
                    "id": account.id,
                    "name": account.name,
                    "utdid": account.utdid,
                    "accessToken": mask_token(account.access_token),
                    "expiresAtText": format_timestamp(account.expires_at),
                    "addedAt": account.added_at,
                },
                "quota": quota,
                "activation": activation,
            }
        )

    @application.get("/v1/models")
    def anthropic_models(request: Request) -> JSONResponse:
        panel_settings = panel_settings_store.load()
        unauthorized = _authorize_proxy_request(request, panel_settings)
        if unauthorized:
            return unauthorized
        catalog, source = _load_dynamic_model_catalog(application, panel_settings)
        payload = (
            build_openai_models_payload_from_catalog(catalog)
            if catalog
            else build_models_payload()
        )
        response = JSONResponse(payload)
        response.headers["x-accio-model-source"] = (
            source if catalog else "static-fallback"
        )
        return response

    @application.get("/models")
    def anthropic_models_compat(request: Request) -> JSONResponse:
        panel_settings = panel_settings_store.load()
        unauthorized = _authorize_proxy_request(request, panel_settings)
        if unauthorized:
            return unauthorized
        catalog, source = _load_dynamic_model_catalog(application, panel_settings)
        payload = (
            build_openai_models_payload_from_catalog(catalog)
            if catalog
            else build_models_payload()
        )
        response = JSONResponse(payload)
        response.headers["x-accio-model-source"] = (
            source if catalog else "static-fallback"
        )
        return response

    @application.get("/v1beta/models")
    def gemini_models(request: Request) -> JSONResponse:
        panel_settings = panel_settings_store.load()
        unauthorized = _authorize_proxy_request(request, panel_settings)
        if unauthorized:
            return JSONResponse(
                gemini_error_payload(401, "无效的 API Key", error_status="UNAUTHENTICATED"),
                status_code=401,
            )
        catalog, source = _load_dynamic_model_catalog(application, panel_settings)
        payload = (
            build_gemini_models_payload_from_catalog(catalog)
            if catalog
            else build_gemini_models_payload()
        )
        response = JSONResponse(payload)
        response.headers["x-accio-model-source"] = (
            source if catalog else "static-fallback"
        )
        return response

    @application.get("/v1beta/models/{model_name}")
    def gemini_model_detail(request: Request, model_name: str) -> JSONResponse:
        panel_settings = panel_settings_store.load()
        unauthorized = _authorize_proxy_request(request, panel_settings)
        if unauthorized:
            return JSONResponse(
                gemini_error_payload(401, "无效的 API Key", error_status="UNAUTHENTICATED"),
                status_code=401,
            )
        model_payload, source = _resolve_gemini_model_payload(
            application,
            panel_settings,
            model_name,
        )
        if model_payload is None:
            available = sorted(_dynamic_gemini_model_names(application, panel_settings))
            return _gemini_error_response(
                404,
                f"未找到模型 {normalize_gemini_model_name(model_name)}。当前可用 Gemini 模型: {', '.join(available)}",
                error_status="NOT_FOUND",
            )
        response = JSONResponse(model_payload)
        response.headers["x-accio-model-source"] = source
        return response

    from .proxy_api_routes import install_proxy_api_routes

    install_proxy_api_routes(application)

    @application.patch("/api/settings")
    def update_settings(
        request: Request,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        current_settings = panel_settings_store.load()
        upstream_proxy_url = str(payload.get("upstreamProxyUrl") or "").strip()
        auto_disable_on_empty_quota = bool(
            payload.get("autoDisableOnEmptyQuota", True)
        )
        auto_enable_on_recovered_quota = bool(
            payload.get("autoEnableOnRecoveredQuota", True)
        )
        api_account_strategy = normalize_api_account_strategy(
            payload.get("apiAccountStrategy")
        )
        admin_password = str(payload.get("adminPassword") or "").strip()

        try:
            normalized_upstream_proxy_url = normalize_upstream_proxy_url(upstream_proxy_url)
        except ValueError as exc:
            return JSONResponse(
                {"success": False, "message": str(exc)},
                status_code=400,
            )

        panel_settings = panel_settings_store.save(
            PanelSettings(
                upstream_proxy_url=normalized_upstream_proxy_url,
                auto_disable_on_empty_quota=auto_disable_on_empty_quota,
                auto_enable_on_recovered_quota=auto_enable_on_recovered_quota,
                api_account_strategy=api_account_strategy,
                admin_password=admin_password or current_settings.admin_password,
                session_secret=current_settings.session_secret,
            )
        )

        return JSONResponse(
            {
                "success": True,
                "message": "设置已保存",
                "settings": panel_settings.to_dict(),
            }
        )

    @application.get("/api/accounts/{account_id}/quota")
    def get_account_quota(request: Request, account_id: str) -> JSONResponse:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        account = store.get_account(account_id)
        if not account:
            return JSONResponse(
                {"success": False, "message": "账号不存在"},
                status_code=404,
            )

        panel_settings = panel_settings_store.load()
        account, quota = _query_quota_with_refresh_fallback(
            store,
            client,
            account,
            panel_settings,
        )

        return JSONResponse(
            {
                "success": quota["success"],
                "message": f"{account.name} 额度已刷新"
                if quota["success"]
                else quota["message"],
                "status": _account_status_view(account),
                "quota": quota,
            }
        )

    @application.get("/auth/callback", response_class=HTMLResponse)
    def auth_callback(
        request: Request,
        accessToken: str | None = Query(default=None),
        refreshToken: str | None = Query(default=None),
        expiresAt: str | None = Query(default=None),
        cookie: str | None = Query(default=None),
    ) -> HTMLResponse:
        if not accessToken or not refreshToken:
            return TEMPLATES.TemplateResponse(
                request=request,
                name="callback.html",
                status_code=400,
                context={
                    "title": "登录失败",
                    "message": "缺少 accessToken 或 refreshToken，无法保存账号。",
                    "account": None,
                    "quota": None,
                },
            )

        panel_settings = panel_settings_store.load()
        account, quota, created, activation = _import_callback_account(
            store,
            client,
            panel_settings,
            access_token=accessToken,
            refresh_token=refreshToken,
            expires_at=expiresAt,
            cookie=cookie,
        )

        return TEMPLATES.TemplateResponse(
            request=request,
            name="callback.html",
            context={
                "title": "登录成功",
                "message": (
                    ("账号已保存到管理面板。" if created else "账号已存在，Token 已更新。")
                    + " "
                    + _activation_summary_text(activation)
                ).strip(),
                "account": {
                    "id": account.id,
                    "name": account.name,
                    "utdid": account.utdid,
                    "access_token": mask_token(account.access_token),
                    "expires_at_text": format_timestamp(account.expires_at),
                    "added_at": account.added_at,
                },
                "quota": quota,
                "activation": activation,
            },
        )

    @application.get("/accounts/{account_id}", response_class=HTMLResponse)
    def account_detail(account_id: str) -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=307)

    @application.post("/api/accounts/{account_id}/refresh")
    def refresh_account(request: Request, account_id: str) -> JSONResponse:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        account = store.get_account(account_id)
        if not account:
            return JSONResponse(
                {"success": False, "message": "账号不存在"},
                status_code=404,
            )

        panel_settings = panel_settings_store.load()
        result = _refresh_token(client, account, panel_settings)
        if not result.get("success"):
            return JSONResponse(
                {
                    "success": False,
                    "message": result.get("message") or "刷新失败",
                },
                status_code=502,
            )

        data = result.get("data") or {}
        updated = store.update_tokens(
            account_id,
            access_token=str(data.get("accessToken") or account.access_token),
            refresh_token=str(data.get("refreshToken") or account.refresh_token),
            expires_at=data.get("expiresAt"),
        )
        if not updated:
            return JSONResponse(
                {"success": False, "message": "账号更新失败"},
                status_code=500,
            )
        if not updated.manual_enabled and updated.auto_disabled_reason:
            updated.manual_enabled = True
            updated.auto_disabled = False
            updated.auto_disabled_reason = None
        updated.next_quota_check_at = _now_timestamp()
        updated.next_quota_check_reason = "Token 刷新后立即检查额度"
        store.save(updated)

        return JSONResponse(
            {
                "success": True,
                "message": f"{updated.name} 刷新成功",
                "expiresAt": updated.expires_at,
            }
        )

    @application.get("/api/accounts/{account_id}/switch", response_model=None)
    def account_switch(request: Request, account_id: str):
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        account = store.get_account(account_id)
        if not account:
            return JSONResponse(
                {"success": False, "message": "账号不存在"},
                status_code=404,
            )

        if not account.access_token:
            return JSONResponse(
                {"success": False, "message": "账号缺少 accessToken"},
                status_code=400,
            )

        panel_settings = panel_settings_store.load()
        callback_url = _effective_callback_url(settings, panel_settings)
        params: dict[str, str] = {
            "accessToken": account.access_token,
        }
        if account.refresh_token:
            params["refreshToken"] = account.refresh_token
        if account.expires_at is not None:
            params["expiresAt"] = str(account.expires_at)
        if account.cookie:
            params["cookie"] = account.cookie
        target_url = f"{callback_url}?{urlencode(params)}"
        return RedirectResponse(target_url, status_code=307)

    @application.patch("/api/accounts/{account_id}/enabled")
    def update_account_enabled(
        request: Request,
        account_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        enabled = payload.get("enabled") if isinstance(payload, dict) else None
        if not isinstance(enabled, bool):
            return JSONResponse(
                {"success": False, "message": "enabled 必须是布尔值"},
                status_code=400,
            )

        updated = store.set_manual_enabled(account_id, enabled)
        if not updated:
            return JSONResponse(
                {"success": False, "message": "账号不存在"},
                status_code=404,
            )

        if enabled and updated.auto_disabled:
            cleared = store.set_auto_disabled(account_id, False, None)
            if cleared:
                updated = cleared
        if enabled and updated.auto_disabled_reason:
            updated.auto_disabled_reason = None
        updated.next_quota_check_at = _now_timestamp() if enabled else None
        updated.next_quota_check_reason = (
            "手动切换启用状态后立即检查额度" if enabled else None
        )
        store.save(updated)

        return JSONResponse(
            {
                "success": True,
                "message": f"{updated.name} 已{'启用' if enabled else '禁用'}",
                "account": {
                    "id": updated.id,
                    "manualEnabled": updated.manual_enabled,
                    "status": _account_status_view(updated),
                    "updatedAt": updated.updated_at,
                },
            }
        )

    @application.patch("/api/accounts/{account_id}")
    def rename_account(
        request: Request,
        account_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        raw_name = payload.get("name") if isinstance(payload, dict) else None
        name = str(raw_name or "").strip()
        if not name:
            return JSONResponse(
                {"success": False, "message": "账号名称不能为空"},
                status_code=400,
            )
        if len(name) > 50:
            return JSONResponse(
                {"success": False, "message": "账号名称不能超过 50 个字符"},
                status_code=400,
            )

        updated = store.rename(account_id, name)
        if not updated:
            return JSONResponse(
                {"success": False, "message": "账号不存在"},
                status_code=404,
            )

        return JSONResponse(
            {
                "success": True,
                "message": f"已重命名为 {updated.name}",
                "account": {
                    "id": updated.id,
                    "name": updated.name,
                    "updatedAt": updated.updated_at,
                },
            }
        )

    @application.patch("/api/accounts/{account_id}/priority")
    def update_account_priority(
        request: Request,
        account_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        raw_priority = payload.get("fillPriority") if isinstance(payload, dict) else None
        try:
            fill_priority = int(str(raw_priority).strip())
        except (AttributeError, TypeError, ValueError):
            return JSONResponse(
                {"success": False, "message": "优先级必须是大于等于 0 的整数"},
                status_code=400,
            )
        if fill_priority < 0:
            return JSONResponse(
                {"success": False, "message": "优先级必须是大于等于 0 的整数"},
                status_code=400,
            )

        updated = store.set_fill_priority(account_id, fill_priority)
        if not updated:
            return JSONResponse(
                {"success": False, "message": "账号不存在"},
                status_code=404,
            )

        return JSONResponse(
            {
                "success": True,
                "message": f"{updated.name} 优先级已更新为 {updated.fill_priority}",
                "account": {
                    "id": updated.id,
                    "fillPriority": updated.fill_priority,
                    "updatedAt": updated.updated_at,
                },
            }
        )

    @application.post("/api/accounts/refresh-all")
    def refresh_all_accounts(request: Request) -> JSONResponse:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        accounts = store.list_accounts()
        panel_settings = panel_settings_store.load()
        success_count = 0
        failures: list[str] = []

        for account in accounts:
            result = _refresh_token(client, account, panel_settings)
            if not result.get("success"):
                failures.append(f"{account.name}: {result.get('message') or '刷新失败'}")
                continue

            data = result.get("data") or {}
            updated = store.update_tokens(
                account.id,
                access_token=str(data.get("accessToken") or account.access_token),
                refresh_token=str(data.get("refreshToken") or account.refresh_token),
                expires_at=data.get("expiresAt"),
            )
            if updated:
                if not updated.manual_enabled and updated.auto_disabled_reason:
                    updated.manual_enabled = True
                    updated.auto_disabled = False
                    updated.auto_disabled_reason = None
                updated.next_quota_check_at = _now_timestamp()
                updated.next_quota_check_reason = "批量刷新 Token 后立即检查额度"
                store.save(updated)
                _query_quota_with_refresh_fallback(
                    store,
                    client,
                    updated,
                    panel_settings,
                )
                success_count += 1
            else:
                failures.append(f"{account.name}: 本地保存失败")

        return JSONResponse(
            {
                "success": not failures,
                "message": f"已刷新 {success_count} 个账号",
                "failures": failures,
            }
        )

    @application.post("/api/accounts/batch")
    def batch_accounts(
        request: Request,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        action = str(payload.get("action") or "").strip()
        raw_account_ids = payload.get("accountIds")
        if not isinstance(raw_account_ids, list):
            return JSONResponse(
                {"success": False, "message": "accountIds 必须是数组"},
                status_code=400,
            )

        account_ids: list[str] = []
        for account_id in raw_account_ids:
            normalized_id = str(account_id or "").strip()
            if normalized_id and normalized_id not in account_ids:
                account_ids.append(normalized_id)

        if not account_ids:
            return JSONResponse(
                {"success": False, "message": "请先选择至少一个账号"},
                status_code=400,
            )

        allowed_actions = {
            "refresh_token",
            "refresh_quota",
            "enable",
            "disable",
            "delete",
        }
        if action not in allowed_actions:
            return JSONResponse(
                {"success": False, "message": "不支持的批量操作"},
                status_code=400,
            )

        panel_settings = panel_settings_store.load()
        success_count = 0
        failures: list[str] = []

        for account_id in account_ids:
            account = store.get_account(account_id)
            if not account:
                failures.append(f"{account_id}: 账号不存在")
                continue

            if action == "refresh_token":
                result = _refresh_token(client, account, panel_settings)
                if not result.get("success"):
                    failures.append(
                        f"{account.name}: {result.get('message') or '刷新失败'}"
                    )
                    continue
                data = result.get("data") or {}
                updated = store.update_tokens(
                    account_id,
                    access_token=str(data.get("accessToken") or account.access_token),
                    refresh_token=str(data.get("refreshToken") or account.refresh_token),
                    expires_at=data.get("expiresAt"),
                )
                if not updated:
                    failures.append(f"{account.name}: 本地保存失败")
                    continue
                if not updated.manual_enabled and updated.auto_disabled_reason:
                    updated.manual_enabled = True
                    updated.auto_disabled = False
                    updated.auto_disabled_reason = None
                updated.next_quota_check_at = _now_timestamp()
                updated.next_quota_check_reason = "批量刷新 Token 后立即检查额度"
                store.save(updated)
                _query_quota_with_refresh_fallback(
                    store,
                    client,
                    updated,
                    panel_settings,
                )
                success_count += 1
                continue

            if action == "refresh_quota":
                _, quota = _query_quota_with_refresh_fallback(
                    store,
                    client,
                    account,
                    panel_settings,
                )
                if not quota["success"]:
                    failures.append(
                        f"{account.name}: {quota['message'] or '额度刷新失败'}"
                    )
                    continue
                success_count += 1
                continue

            if action == "enable":
                updated = store.set_manual_enabled(account_id, True)
                if not updated:
                    failures.append(f"{account.name}: 启用失败")
                    continue
                if updated.auto_disabled:
                    cleared = store.set_auto_disabled(account_id, False, None)
                    if cleared:
                        updated = cleared
                if updated.auto_disabled_reason:
                    updated.auto_disabled_reason = None
                updated.next_quota_check_at = _now_timestamp()
                updated.next_quota_check_reason = "批量启用后立即检查额度"
                store.save(updated)
                success_count += 1
                continue

            if action == "disable":
                updated = store.set_manual_enabled(account_id, False)
                if not updated:
                    failures.append(f"{account.name}: 禁用失败")
                    continue
                updated.next_quota_check_at = None
                updated.next_quota_check_reason = None
                store.save(updated)
                success_count += 1
                continue

            if not store.delete(account_id):
                failures.append(f"{account.name}: 删除失败")
                continue
            success_count += 1

        action_labels = {
            "refresh_token": "刷新 Token",
            "refresh_quota": "刷新额度",
            "enable": "启用",
            "disable": "禁用",
            "delete": "删除",
        }
        return JSONResponse(
            {
                "success": not failures,
                "message": f"已批量{action_labels[action]} {success_count} 个账号",
                "processedCount": success_count,
                "failureCount": len(failures),
                "failures": failures,
            }
        )

    @application.post("/api/accounts/import")
    def import_accounts(
        request: Request,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        raw_files = payload.get("files")
        if not isinstance(raw_files, list):
            return JSONResponse(
                {"success": False, "message": "files 必须是数组"},
                status_code=400,
            )

        account_payloads: list[dict[str, Any]] = []
        failures: list[str] = []

        for index, item in enumerate(raw_files, start=1):
            if not isinstance(item, dict):
                failures.append(f"第 {index} 个文件格式无效")
                continue

            file_name = str(item.get("name") or f"第 {index} 个文件").strip()
            content = str(item.get("content") or "").strip()
            if not content:
                failures.append(f"{file_name}: 文件内容为空")
                continue

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                failures.append(f"{file_name}: 不是合法的 JSON")
                continue

            if isinstance(parsed, dict):
                account_payloads.append(parsed)
                continue

            if isinstance(parsed, list):
                valid_items = 0
                for item_index, account_payload in enumerate(parsed, start=1):
                    if not isinstance(account_payload, dict):
                        failures.append(
                            f"{file_name}: 第 {item_index} 项不是账号对象"
                        )
                        continue
                    account_payloads.append(account_payload)
                    valid_items += 1
                if valid_items == 0:
                    failures.append(f"{file_name}: 没有可导入的账号数据")
                continue

            failures.append(f"{file_name}: 仅支持账号对象或账号数组")

        if not account_payloads and failures:
            return JSONResponse(
                {
                    "success": False,
                    "message": "导入失败，没有可用的账号数据",
                    "createdCount": 0,
                    "updatedCount": 0,
                    "failureCount": len(failures),
                    "importedCount": 0,
                    "failures": failures,
                },
                status_code=400,
            )

        if not account_payloads:
            return JSONResponse(
                {"success": False, "message": "请至少选择一个账号 JSON 文件"},
                status_code=400,
            )

        result = store.import_accounts(account_payloads)
        merged_failures = failures + result["failures"]
        imported_count = result["importedCount"]
        created_count = result["createdCount"]
        updated_count = result["updatedCount"]
        failure_count = len(merged_failures)

        if imported_count > 0:
            message = (
                f"已导入 {imported_count} 个账号"
                f"（新增 {created_count}，更新 {updated_count}）"
            )
        else:
            message = "没有成功导入任何账号"

        return JSONResponse(
            {
                "success": failure_count == 0,
                "message": message,
                "createdCount": created_count,
                "updatedCount": updated_count,
                "failureCount": failure_count,
                "importedCount": imported_count,
                "failures": merged_failures,
            }
        )

    @application.get("/api/accounts/{account_id}/detail")
    def account_detail_data(request: Request, account_id: str) -> JSONResponse:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        account = store.get_account(account_id)
        if not account:
            return JSONResponse(
                {"success": False, "message": "账号不存在"},
                status_code=404,
            )

        panel_settings = panel_settings_store.load()
        account, quota = _query_quota_with_refresh_fallback(
            store,
            client,
            account,
            panel_settings,
        )
        return JSONResponse(
            {
                "success": True,
                "account": {
                    "id": account.id,
                    "name": account.name,
                    "utdid": account.utdid,
                    "fillPriority": account.fill_priority,
                    "accessToken": account.access_token,
                    "refreshToken": account.refresh_token,
                    "expiresAtText": format_timestamp(account.expires_at),
                    "addedAt": account.added_at,
                    "updatedAt": account.updated_at,
                    "lastQuotaCheckAt": format_timestamp(account.last_quota_check_at),
                    "nextQuotaCheckAt": format_timestamp(account.next_quota_check_at),
                    "nextQuotaCheckReason": account.next_quota_check_reason or "-",
                    "disabledModels": _disabled_model_items(account),
                    "status": _account_status_view(account),
                },
                "quota": quota,
            }
        )

    @application.get("/api/accounts/{account_id}/download")
    def download_account_json(request: Request, account_id: str) -> Response:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        account = store.get_account(account_id)
        if not account:
            return JSONResponse(
                {"success": False, "message": "账号不存在"},
                status_code=404,
            )

        content = json.dumps(account.to_dict(), ensure_ascii=False, indent=2)
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{account.id}.json"'
            },
        )

    @application.delete("/api/accounts/{account_id}")
    def delete_account(request: Request, account_id: str) -> JSONResponse:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        deleted = store.delete(account_id)
        if not deleted:
            return JSONResponse(
                {"success": False, "message": "账号不存在"},
                status_code=404,
            )
        return JSONResponse({"success": True, "message": "账号已删除"})

    return application


app = create_app()


def run() -> None:
    settings: Settings = app.state.settings
    panel_settings_store: PanelSettingsStore = app.state.panel_settings_store
    panel_settings = panel_settings_store.load()
    effective_api_base_url = _effective_api_base_url(
        settings,
        panel_settings,
    )

    print("=" * 56)
    print(" Accio 多账号管理面板")
    print("=" * 56)
    print(f"管理面板: http://{settings.callback_host}:{settings.callback_port}/dashboard")
    print(f"OAuth 页面: http://{settings.callback_host}:{settings.callback_port}/oauth")
    print(f"本地回调: {settings.callback_url}")
    print(f"Anthropic API: {effective_api_base_url}/v1/messages")
    print(f"OpenAI Chat API: {effective_api_base_url}/v1/chat/completions")
    print(f"OpenAI Responses API: {effective_api_base_url}/v1/responses")
    print(f"模型列表: {effective_api_base_url}/v1/models")
    print(f"Gemini API: {effective_api_base_url}/v1beta/models/{{model}}:generateContent")
    print(f"Gemini 模型列表: {effective_api_base_url}/v1beta/models")
    print(f"API 调度: {_api_account_strategy_label(panel_settings.api_account_strategy)}")
    print(f"上游代理: {panel_settings.upstream_proxy_url or '未配置'}")
    print(f"配置存储: {'MySQL' if settings.database_enabled else '本地文件'}")
    if settings.database_enabled:
        print(f"MySQL: {settings.database_summary}")
    else:
        print(f"账号目录: {settings.accounts_dir}")
        print(f"旧版迁移源: {settings.accounts_file}")
        print(f"配置文件: {settings.settings_file}")
    print(f"统计文件: {settings.stats_file}")
    print(f"日志文件: {settings.api_logs_file}")
    print("=" * 56)

    if settings.auto_open_browser:
        dashboard_url = (
            f"http://{settings.callback_host}:{settings.callback_port}/dashboard"
        )
        threading.Timer(1, lambda: webbrowser.open(dashboard_url)).start()

    uvicorn.run(app, host=settings.server_host, port=settings.callback_port)
