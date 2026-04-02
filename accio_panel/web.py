from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .app_settings import (
    PanelSettings,
    PanelSettingsStore,
    normalize_api_account_strategy,
    normalize_public_base_url,
    normalize_upstream_proxy_url,
)
from .anthropic_proxy import (
    DEFAULT_ANTHROPIC_MODEL,
    SUPPORTED_ANTHROPIC_MODELS,
    SUPPORTED_ANTHROPIC_MODELS_SET,
    anthropic_error_payload,
    build_accio_request,
    build_models_payload,
    decode_non_stream_response,
    iter_anthropic_sse_bytes,
)
from .client import AccioClient
from .config import Settings
from .models import Account
from .store import AccountStore
from .usage_stats import UsageStatsStore
from .utils import format_countdown_hours, format_timestamp, mask_token


TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

ENABLED_ACCOUNT_CHECK_INTERVAL_SECONDS = 15 * 60
FAILED_ACCOUNT_RETRY_SECONDS = 5 * 60
RECOVERY_CHECK_BUFFER_SECONDS = 90
SCHEDULER_TICK_SECONDS = 30
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


def _is_admin_authenticated(request: Request) -> bool:
    return bool(request.session.get("admin_authenticated"))


def _effective_callback_url(settings: Settings, panel_settings: PanelSettings) -> str:
    return panel_settings.effective_callback_url(_local_base_url(settings))


def _effective_api_base_url(settings: Settings, panel_settings: PanelSettings) -> str:
    return panel_settings.effective_base_url(_local_base_url(settings))


def _now_timestamp() -> int:
    return int(time.time())


def _api_account_strategy_label(strategy: str) -> str:
    return "轮询" if strategy == "round_robin" else "优先填充"


def _account_status_view(account: Account) -> dict[str, Any]:
    if not account.manual_enabled:
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


def _extract_proxy_api_key(request: Request) -> str:
    api_key = str(request.headers.get("x-api-key") or "").strip()
    if api_key:
        return api_key

    authorization = str(request.headers.get("authorization") or "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
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


def _ordered_proxy_candidates(store: AccountStore) -> list[Account]:
    return [account for account in store.list_accounts() if account.manual_enabled]


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


def _check_proxy_candidate(
    store: AccountStore,
    client: AccioClient,
    panel_settings: PanelSettings,
    account: Account,
) -> tuple[Account, dict[str, Any]]:
    return _apply_quota_result(
        store,
        account,
        _query_quota(client, account, panel_settings),
        panel_settings,
    )


def _select_proxy_account(
    application: FastAPI,
    panel_settings: PanelSettings,
) -> tuple[Account, dict[str, Any]]:
    store: AccountStore = application.state.store
    client: AccioClient = application.state.client

    candidates = _ordered_proxy_candidates(store)
    if not candidates:
        raise ProxySelectionError(503, "当前没有已启用的账号可供 API 调用。")

    strategy = panel_settings.api_account_strategy
    if strategy == "round_robin":
        start_index = application.state.proxy_round_robin_index % len(candidates)
        index_order = [
            (start_index + offset) % len(candidates)
            for offset in range(len(candidates))
        ]
    else:
        index_order = list(range(len(candidates)))

    errors: list[str] = []
    for index in index_order:
        account, quota = _check_proxy_candidate(
            store,
            client,
            panel_settings,
            candidates[index],
        )
        if account.auto_disabled:
            errors.append(
                f"{account.name}: {account.auto_disabled_reason or '账号已自动禁用。'}"
            )
            continue
        if not quota["success"]:
            errors.append(f"{account.name}: {quota['message'] or '额度查询失败'}")
            continue
        if quota["remaining_value"] <= 0:
            errors.append(f"{account.name}: 剩余额度为 0%")
            continue

        if strategy == "round_robin":
            application.state.proxy_round_robin_index = (index + 1) % len(candidates)
        return account, quota

    if strategy == "round_robin":
        application.state.proxy_round_robin_index = 0

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


def _build_quota_view(result: dict[str, Any]) -> dict[str, Any]:
    data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(data, dict):
        data = {}

    usage = data.get("usagePercent")
    countdown = data.get("refreshCountdownSeconds")
    usage_value = 0

    try:
        usage_value = int(float(usage))
    except (TypeError, ValueError):
        usage_value = 0
    usage_value = max(0, min(usage_value, 100))
    remaining_value = max(0, 100 - usage_value)

    if result.get("success"):
        level = "low"
        if remaining_value < 20:
            level = "high"
        elif remaining_value < 50:
            level = "medium"
        return {
            "success": True,
            "used_value": usage_value,
            "used_text": f"{usage_value}%",
            "remaining_value": remaining_value,
            "remaining_text": f"{remaining_value}%",
            "reset_text": format_countdown_hours(countdown),
            "level": level,
            "message": _normalize_success_message(result.get("message")),
        }

    return {
        "success": False,
        "used_value": 0,
        "used_text": "-",
        "remaining_value": 0,
        "remaining_text": "获取失败",
        "reset_text": "-",
        "level": "error",
        "message": result.get("message") or "额度查询失败",
    }


def _extract_countdown_seconds(result: dict[str, Any]) -> int | None:
    data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(data, dict):
        return None

    raw_countdown = data.get("refreshCountdownSeconds")
    try:
        countdown = int(float(raw_countdown))
    except (TypeError, ValueError):
        return None
    return max(0, countdown)


def _plan_next_quota_check(
    account: Account,
    *,
    quota_success: bool,
    countdown_seconds: int | None,
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
        if countdown_seconds is not None:
            return (
                now_ts + countdown_seconds + RECOVERY_CHECK_BUFFER_SECONDS,
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
    countdown_seconds = _extract_countdown_seconds(quota_result)
    now_ts = _now_timestamp()
    should_mark_updated = False

    if quota["success"]:
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
        countdown_seconds=countdown_seconds,
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

    quota_map: dict[str, dict[str, Any]] = {}
    workers = min(8, len(accounts))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_query_quota, client, account, panel_settings): account.id
            for account in accounts
        }
        for future in as_completed(future_map):
            account_id = future_map[future]
            try:
                quota_map[account_id] = future.result()
            except Exception as exc:  # pragma: no cover
                quota_map[account_id] = {"success": False, "message": str(exc)}

    items: list[dict[str, Any]] = []
    for account in accounts:
        account, quota_view = _apply_quota_result(
            store,
            account,
            quota_map.get(account.id, {}),
            panel_settings,
        )
        status = _account_status_view(account)
        items.append(
            {
                "id": account.id,
                "name": account.name,
                "utdid": account.utdid,
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
        for account in accounts:
            if not account.manual_enabled:
                if (
                    account.next_quota_check_at is not None
                    or account.next_quota_check_reason is not None
                ):
                    account.next_quota_check_at = None
                    account.next_quota_check_reason = None
                    store.save(account)
                continue

            if account.next_quota_check_at is None or account.next_quota_check_at <= now_ts:
                due_accounts.append(account)

        for account in due_accounts:
            quota_result = await asyncio.to_thread(
                _query_quota,
                client,
                account,
                panel_settings,
            )
            _apply_quota_result(store, account, quota_result, panel_settings)

        await asyncio.sleep(SCHEDULER_TICK_SECONDS)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store = AccountStore(settings.accounts_dir, settings.accounts_file)
    client = AccioClient(settings)
    usage_stats_store = UsageStatsStore(settings.stats_file)
    panel_settings_store = PanelSettingsStore(
        settings.settings_file,
        settings.legacy_settings_file,
    )
    initial_panel_settings = panel_settings_store.load()

    application = FastAPI(
        title="Accio 多账号管理面板",
        description="支持 Token 刷新、额度查看、登录链接获取与 FastAPI 回调保存。",
        version=settings.version,
    )
    application.add_middleware(
        SessionMiddleware,
        secret_key=initial_panel_settings.session_secret,
        session_cookie="accio_admin_session",
        same_site="lax",
        max_age=60 * 60 * 24 * 7,
        https_only=False,
    )
    application.state.settings = settings
    application.state.store = store
    application.state.client = client
    application.state.usage_stats_store = usage_stats_store
    application.state.panel_settings_store = panel_settings_store
    application.state.quota_scheduler_task = None
    application.state.proxy_round_robin_index = 0

    @application.on_event("startup")
    async def startup_scheduler() -> None:
        task = application.state.quota_scheduler_task
        if task is None or task.done():
            application.state.quota_scheduler_task = asyncio.create_task(
                _quota_scheduler_loop(application)
            )

    @application.on_event("shutdown")
    async def shutdown_scheduler() -> None:
        task = application.state.quota_scheduler_task
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        application.state.quota_scheduler_task = None

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
                    "login_url": client.build_login_url(callback_url),
                },
            )

        current_view = _parse_dashboard_view(request.query_params.get("view"))
        page_size = _parse_page_size(request.query_params.get("pageSize"))
        requested_page = _parse_page_number(request.query_params.get("page"))
        all_accounts = store.list_accounts()
        account_count = len(all_accounts)
        total_pages = max(1, ((account_count - 1) // page_size) + 1) if account_count else 1
        current_page = min(requested_page, total_pages)
        page_start = (current_page - 1) * page_size
        page_end = page_start + page_size
        page_accounts = all_accounts[page_start:page_end] if current_view == "accounts" else []
        stats_snapshot = usage_stats_store.snapshot(
            {account.id: account.name for account in all_accounts}
        )
        dashboard_items = _build_dashboard_items(
            page_accounts,
            client,
            store,
            panel_settings,
        )
        api_base_url = _effective_api_base_url(settings, panel_settings)
        return TEMPLATES.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "page_title": "Accio 多账号管理面板",
                "accounts": dashboard_items,
                "account_count": account_count,
                "callback_url": callback_url,
                "login_url": client.build_login_url(callback_url),
                "public_base_url": panel_settings.public_base_url,
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
                "supported_models_text": " / ".join(SUPPORTED_ANTHROPIC_MODELS),
                "current_view": current_view,
                "current_page": current_page,
                "page_size": page_size,
                "page_size_options": PAGE_SIZE_OPTIONS,
                "total_pages": total_pages,
                "page_numbers": _build_page_numbers(current_page, total_pages),
                "page_start_index": page_start + 1 if page_accounts else 0,
                "page_end_index": page_start + len(page_accounts),
                "usage_stats": stats_snapshot,
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

    @application.get("/v1/models")
    def anthropic_models(request: Request) -> JSONResponse:
        panel_settings = panel_settings_store.load()
        unauthorized = _authorize_proxy_request(request, panel_settings)
        if unauthorized:
            return unauthorized
        return JSONResponse(build_models_payload())

    @application.get("/models")
    def anthropic_models_compat(request: Request) -> JSONResponse:
        panel_settings = panel_settings_store.load()
        unauthorized = _authorize_proxy_request(request, panel_settings)
        if unauthorized:
            return unauthorized
        return JSONResponse(build_models_payload())

    @application.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        panel_settings = panel_settings_store.load()
        usage_stats_store: UsageStatsStore = application.state.usage_stats_store
        unauthorized = _authorize_proxy_request(request, panel_settings)
        if unauthorized:
            return unauthorized

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
        if model not in SUPPORTED_ANTHROPIC_MODELS_SET:
            return _anthropic_error_response(
                400,
                f"不支持模型 {model}。当前仅支持: {', '.join(SUPPORTED_ANTHROPIC_MODELS)}",
                error_type="invalid_request_error",
            )

        try:
            account, quota = await asyncio.to_thread(
                _select_proxy_account,
                application,
                panel_settings,
            )
        except ProxySelectionError as exc:
            return _anthropic_error_response(exc.status_code, exc.message)

        accio_body = build_accio_request(
            payload,
            token=account.access_token,
            utdid=account.utdid,
            version=settings.version,
        )

        try:
            upstream_response = await asyncio.to_thread(
                client.generate_content,
                account,
                accio_body,
                proxy_url=panel_settings.upstream_proxy_url,
            )
        except requests.RequestException as exc:
            usage_stats_store.record_message(
                account_id=account.id,
                model=model,
                input_tokens=0,
                output_tokens=0,
                success=False,
                stop_reason="request_exception",
            )
            return _anthropic_error_response(502, f"上游请求失败: {exc}")

        response_headers = {
            "x-accio-account-id": account.id,
            "x-accio-account-strategy": panel_settings.api_account_strategy,
        }
        if quota.get("success"):
            response_headers["x-accio-account-remaining"] = str(
                quota["remaining_value"]
            )

        if not upstream_response.ok:
            try:
                upstream_text = upstream_response.text[:500]
            finally:
                upstream_response.close()
            usage_stats_store.record_message(
                account_id=account.id,
                model=model,
                input_tokens=0,
                output_tokens=0,
                success=False,
                stop_reason="upstream_error",
            )
            return _anthropic_error_response(
                upstream_response.status_code,
                upstream_text or "上游返回错误。",
            )

        if payload.get("stream", True) is not False:
            def record_stream_summary(summary: dict[str, Any]) -> None:
                usage = summary.get("usage") if isinstance(summary, dict) else {}
                if not isinstance(usage, dict):
                    usage = {}
                usage_stats_store.record_message(
                    account_id=account.id,
                    model=model,
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    success=True,
                    stop_reason=str(summary.get("stop_reason") or "end_turn"),
                )

            return StreamingResponse(
                iter_anthropic_sse_bytes(
                    upstream_response,
                    model,
                    on_complete=record_stream_summary,
                ),
                media_type="text/event-stream",
                headers=response_headers,
            )

        response_payload = await asyncio.to_thread(
            decode_non_stream_response,
            upstream_response,
            model,
        )
        usage = response_payload.get("usage") if isinstance(response_payload, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        usage_stats_store.record_message(
            account_id=account.id,
            model=model,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            success=True,
            stop_reason=str(response_payload.get("stop_reason") or "end_turn"),
        )
        return JSONResponse(response_payload, headers=response_headers)

    @application.patch("/api/settings")
    def update_settings(
        request: Request,
        payload: dict[str, Any] = Body(...),
    ) -> JSONResponse:
        if not _is_admin_authenticated(request):
            return _unauthorized_json()

        current_settings = panel_settings_store.load()
        public_base_url = str(payload.get("publicBaseUrl") or "").strip()
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
            normalized_public_base_url = normalize_public_base_url(public_base_url)
            normalized_upstream_proxy_url = normalize_upstream_proxy_url(upstream_proxy_url)
        except ValueError as exc:
            return JSONResponse(
                {"success": False, "message": str(exc)},
                status_code=400,
            )

        panel_settings = panel_settings_store.save(
            PanelSettings(
                public_base_url=normalized_public_base_url,
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
                "callbackUrl": _effective_callback_url(settings, panel_settings),
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
        account, quota = _apply_quota_result(
            store,
            account,
            _query_quota(client, account, panel_settings),
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

        # 登录回调优先做 upsert，避免同一账号重复写入本地列表。
        account, created = store.upsert_from_callback(
            access_token=accessToken,
            refresh_token=refreshToken,
            expires_at=expiresAt,
            cookie=cookie,
        )
        panel_settings = panel_settings_store.load()
        account, quota = _apply_quota_result(
            store,
            account,
            _query_quota(client, account, panel_settings),
            panel_settings,
        )

        return TEMPLATES.TemplateResponse(
            request=request,
            name="callback.html",
            context={
                "title": "登录成功",
                "message": "账号已保存到管理面板。" if created else "账号已存在，Token 已更新。",
                "account": {
                    "id": account.id,
                    "name": account.name,
                    "utdid": account.utdid,
                    "access_token": mask_token(account.access_token),
                    "expires_at_text": format_timestamp(account.expires_at),
                    "added_at": account.added_at,
                },
                "quota": quota,
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
                updated.next_quota_check_at = _now_timestamp()
                updated.next_quota_check_reason = "批量刷新 Token 后立即检查额度"
                store.save(updated)
                _apply_quota_result(
                    store,
                    updated,
                    _query_quota(client, updated, panel_settings),
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
                updated.next_quota_check_at = _now_timestamp()
                updated.next_quota_check_reason = "批量刷新 Token 后立即检查额度"
                store.save(updated)
                _apply_quota_result(
                    store,
                    updated,
                    _query_quota(client, updated, panel_settings),
                    panel_settings,
                )
                success_count += 1
                continue

            if action == "refresh_quota":
                _, quota = _apply_quota_result(
                    store,
                    account,
                    _query_quota(client, account, panel_settings),
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
        account, quota = _apply_quota_result(
            store,
            account,
            _query_quota(client, account, panel_settings),
            panel_settings,
        )
        return JSONResponse(
            {
                "success": True,
                "account": {
                    "id": account.id,
                    "name": account.name,
                    "utdid": account.utdid,
                    "accessToken": account.access_token,
                    "refreshToken": account.refresh_token,
                    "expiresAtText": format_timestamp(account.expires_at),
                    "addedAt": account.added_at,
                    "updatedAt": account.updated_at,
                    "lastQuotaCheckAt": format_timestamp(account.last_quota_check_at),
                    "nextQuotaCheckAt": format_timestamp(account.next_quota_check_at),
                    "nextQuotaCheckReason": account.next_quota_check_reason or "-",
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
    effective_callback_url = _effective_callback_url(
        settings,
        panel_settings,
    )
    effective_api_base_url = _effective_api_base_url(
        settings,
        panel_settings,
    )

    print("=" * 56)
    print(" Accio 多账号管理面板")
    print("=" * 56)
    print(f"管理面板: http://{settings.callback_host}:{settings.callback_port}/dashboard")
    print(f"本地回调: {settings.callback_url}")
    print(f"当前回调: {effective_callback_url}")
    print(f"Anthropic API: {effective_api_base_url}/v1/messages")
    print(f"模型列表: {effective_api_base_url}/v1/models")
    print(f"API 调度: {_api_account_strategy_label(panel_settings.api_account_strategy)}")
    print(f"上游代理: {panel_settings.upstream_proxy_url or '未配置'}")
    print(f"账号目录: {settings.accounts_dir}")
    print(f"统计文件: {settings.stats_file}")
    print(f"旧版迁移源: {settings.accounts_file}")
    print(f"配置文件: {settings.settings_file}")
    print("=" * 56)

    if settings.auto_open_browser:
        dashboard_url = (
            f"http://{settings.callback_host}:{settings.callback_port}/dashboard"
        )
        threading.Timer(1, lambda: webbrowser.open(dashboard_url)).start()

    uvicorn.run(app, host=settings.callback_host, port=settings.callback_port)
