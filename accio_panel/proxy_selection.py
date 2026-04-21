from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlparse

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .anthropic_proxy import anthropic_error_payload
from .app_settings import PanelSettings
from .client import AccioClient
from .config import Settings
from .gemini_proxy import gemini_error_payload, normalize_gemini_model_name
from .models import Account
from .openai_proxy import openai_error_payload
from .store import AccountStore
from .utils import format_timestamp, mask_token


ENABLED_ACCOUNT_CHECK_INTERVAL_SECONDS = 15 * 60
FAILED_ACCOUNT_RETRY_SECONDS = 5 * 60
RECOVERY_CHECK_BUFFER_SECONDS = 90
ABNORMAL_ACCOUNT_RECOVERY_INTERVAL_SECONDS = 30 * 60
UPSTREAM_QUOTA_EXHAUSTED_MAX_WAIT_SECONDS = 30 * 60
UPSTREAM_QUOTA_EXHAUSTED_AUTO_DISABLED_REASON = (
    "上游返回 [429]: quota exhausted，系统已暂时跳过该账号并等待自动恢复重试。"
)
UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON = "上游 quota exhausted 后自动恢复重试"


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


def _is_upstream_quota_exhausted_cooldown(account: Account) -> bool:
    return (
        account.auto_disabled
        and str(account.next_quota_check_reason or "").strip()
        == UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON
    )


def _mark_account_quota_exhausted_cooldown(
    store: AccountStore,
    account: Account,
) -> Account:
    now_ts = _now_timestamp()
    updated = store.get_account(account.id) or account
    updated.auto_disabled = True
    updated.auto_disabled_reason = UPSTREAM_QUOTA_EXHAUSTED_AUTO_DISABLED_REASON
    updated.last_quota_check_at = now_ts
    updated.next_quota_check_at = now_ts + FAILED_ACCOUNT_RETRY_SECONDS
    updated.next_quota_check_reason = UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON
    updated.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))
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
        if _is_upstream_quota_exhausted_cooldown(account):
            if next_billing_at is not None:
                capped_retry_at = min(
                    next_billing_at + RECOVERY_CHECK_BUFFER_SECONDS,
                    now_ts + UPSTREAM_QUOTA_EXHAUSTED_MAX_WAIT_SECONDS,
                )
                return (
                    max(now_ts + RECOVERY_CHECK_BUFFER_SECONDS, capped_retry_at),
                    UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON,
                )
            return (
                now_ts + FAILED_ACCOUNT_RETRY_SECONDS,
                UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON,
            )
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
    force_auto_recover = _is_upstream_quota_exhausted_cooldown(account)

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
            account.auto_disabled
            and quota["remaining_value"] > 0
            and (panel_settings.auto_enable_on_recovered_quota or force_auto_recover)
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
