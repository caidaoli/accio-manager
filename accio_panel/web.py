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
from .dashboard_views import (
    DEFAULT_PAGE_SIZE,
    PAGE_SIZE_OPTIONS,
    _build_dashboard_items,
    _build_page_numbers,
    _parse_dashboard_view,
    _parse_page_number,
    _parse_page_size,
)
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
    build_gemini_models_payload_from_catalog,
    build_openai_models_payload_from_catalog,
)
from .model_catalog_cache import (
    MODEL_CATALOG_CACHE_SECONDS,
    _dynamic_gemini_model_names,
    _dynamic_proxy_model_names,
    _initial_model_catalog_cache,
    _is_allowed_dynamic_model,
    _load_dynamic_model_catalog,
    _model_catalog_dashboard_text,
    _resolve_gemini_model_payload,
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
from .proxy_selection import (
    ABNORMAL_ACCOUNT_RECOVERY_INTERVAL_SECONDS,
    ENABLED_ACCOUNT_CHECK_INTERVAL_SECONDS,
    FAILED_ACCOUNT_RETRY_SECONDS,
    RECOVERY_CHECK_BUFFER_SECONDS,
    UPSTREAM_QUOTA_EXHAUSTED_AUTO_DISABLED_REASON,
    UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON,
    ProxySelectionError,
    _account_model_disabled_reason,
    _account_status_view,
    _activate_callback_account,
    _activation_summary_text,
    _anthropic_error_response,
    _api_account_strategy_label,
    _apply_quota_result,
    _as_int,
    _authorize_proxy_request,
    _build_quota_view,
    _cached_quota_view,
    _check_proxy_candidate,
    _disable_account_after_refresh_failure,
    _disable_account_model_on_empty_response,
    _disabled_model_items,
    _effective_api_base_url,
    _effective_callback_url,
    _empty_response_log_message,
    _extract_next_billing_timestamp,
    _extract_proxy_api_key,
    _extract_subscription_entitlement,
    _gemini_error_response,
    _import_callback_account,
    _is_admin_authenticated,
    _is_upstream_quota_exhausted_cooldown,
    _iter_upstream_sse_bytes,
    _local_base_url,
    _mark_account_quota_exhausted_cooldown,
    _native_error_response,
    _normalize_success_message,
    _normalize_target_model,
    _now_timestamp,
    _openai_error_response,
    _ordered_proxy_candidates,
    _parse_billing_timestamp,
    _parse_callback_payload,
    _plan_next_quota_check,
    _proxy_fill_sort_key,
    _query_llm_config_with_refresh_fallback,
    _query_quota,
    _query_quota_with_refresh_fallback,
    _refresh_token,
    _request_base_url,
    _select_proxy_account,
    _should_disable_model_on_empty_response,
    _sorted_enabled_accounts,
    _try_recover_abnormal_account,
    _unauthorized_json,
)
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

SCHEDULER_TICK_SECONDS = 30

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
