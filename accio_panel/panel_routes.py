from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode

from fastapi import Body, FastAPI, Query, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.templating import Jinja2Templates

from .anthropic_proxy import build_models_payload
from .app_settings import (
    PanelSettings,
    PanelSettingsStore,
    normalize_api_account_strategy,
    normalize_upstream_proxy_url,
)
from .api_logs import ApiLogStore
from .client import AccioClient
from .config import Settings
from .dashboard_views import (
    PAGE_SIZE_OPTIONS,
    _build_dashboard_items,
    _build_page_numbers,
    _parse_dashboard_view,
    _parse_page_number,
    _parse_page_size,
)
from .gemini_proxy import (
    build_gemini_models_payload,
    gemini_error_payload,
    normalize_gemini_model_name,
)
from .model_catalog import (
    build_gemini_models_payload_from_catalog,
    build_openai_models_payload_from_catalog,
)
from .model_catalog_cache import (
    _dynamic_gemini_model_names,
    _load_dynamic_model_catalog,
    _model_catalog_dashboard_text,
    _resolve_gemini_model_payload,
)
from .proxy_selection import (
    _account_status_view,
    _activation_summary_text,
    _api_account_strategy_label,
    _authorize_proxy_request,
    _disabled_model_items,
    _effective_callback_url,
    _gemini_error_response,
    _import_callback_account,
    _is_admin_authenticated,
    _now_timestamp,
    _parse_callback_payload,
    _query_quota_with_refresh_fallback,
    _refresh_token,
    _request_base_url,
    _unauthorized_json,
)
from .store import AccountStore
from .usage_stats import UsageStatsStore
from .utils import format_timestamp, mask_token

_TEMPLATES = Jinja2Templates(directory=__import__("pathlib").Path(__file__).parent / "templates")

def register_panel_routes(
    application: FastAPI,
    settings: Settings,
    store: AccountStore,
    client: AccioClient,
    panel_settings_store: PanelSettingsStore,
    usage_stats_store: UsageStatsStore,
    api_log_store: ApiLogStore,
) -> None:
    """Register all panel management routes on the given FastAPI application."""

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
            return _TEMPLATES.TemplateResponse(
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
        return _TEMPLATES.TemplateResponse(
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
        return _TEMPLATES.TemplateResponse(
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
            return _TEMPLATES.TemplateResponse(
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

        return _TEMPLATES.TemplateResponse(
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

