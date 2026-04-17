from __future__ import annotations

import asyncio
import contextlib
import threading
import webbrowser

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from .api_logs import ApiLogStore
from .app_settings import PanelSettingsStore
from .client import AccioClient
from .config import Settings
from .model_catalog_cache import _initial_model_catalog_cache
from .persistence import create_runtime_stores
from .proxy_selection import (
    _api_account_strategy_label,
    _effective_api_base_url,
)
from .quota_scheduler import _quota_scheduler_loop
from .usage_stats import UsageStatsStore


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

    from .panel_routes import register_panel_routes

    register_panel_routes(
        application,
        settings,
        store,
        client,
        panel_settings_store,
        usage_stats_store,
        api_log_store,
    )

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
