from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from .. import web as web_module
from ..api_logs import ApiLogStore
from ..app_settings import PanelSettingsStore
from ..client import AccioClient
from ..config import Settings
from ..store import AccountStore
from ..usage_stats import UsageStatsStore


@dataclass(slots=True)
class ProxyRouteContext:
    application: FastAPI
    settings: Settings
    store: AccountStore
    client: AccioClient
    panel_settings_store: PanelSettingsStore
    usage_stats_store: UsageStatsStore
    api_log_store: ApiLogStore

    @classmethod
    def from_application(cls, application: FastAPI) -> "ProxyRouteContext":
        return cls(
            application=application,
            settings=application.state.settings,
            store=application.state.store,
            client=application.state.client,
            panel_settings_store=application.state.panel_settings_store,
            usage_stats_store=application.state.usage_stats_store,
            api_log_store=application.state.api_log_store,
        )

    @property
    def helpers(self):
        return web_module

    @property
    def ProxySelectionError(self):
        return self.helpers.ProxySelectionError

    def authorize_proxy_request(self, *args: Any, **kwargs: Any):
        return self.helpers._authorize_proxy_request(*args, **kwargs)

    def is_allowed_dynamic_model(self, *args: Any, **kwargs: Any):
        return self.helpers._is_allowed_dynamic_model(*args, **kwargs)

    def select_proxy_account(self, *args: Any, **kwargs: Any):
        return self.helpers._select_proxy_account(*args, **kwargs)

    def empty_response_log_message(self, *args: Any, **kwargs: Any):
        return self.helpers._empty_response_log_message(*args, **kwargs)

    def should_disable_model_on_empty_response(self, *args: Any, **kwargs: Any):
        return self.helpers._should_disable_model_on_empty_response(*args, **kwargs)

    def disable_account_model_on_empty_response(self, *args: Any, **kwargs: Any):
        return self.helpers._disable_account_model_on_empty_response(*args, **kwargs)

    def extract_proxy_api_key(self, *args: Any, **kwargs: Any):
        return self.helpers._extract_proxy_api_key(*args, **kwargs)

    def iter_upstream_sse_bytes(self, *args: Any, **kwargs: Any):
        return self.helpers._iter_upstream_sse_bytes(*args, **kwargs)

    def gemini_error_response(self, *args: Any, **kwargs: Any):
        return self.helpers._gemini_error_response(*args, **kwargs)

    def decode_gemini_generate_content_response(self, *args: Any, **kwargs: Any):
        return self.helpers.decode_gemini_generate_content_response(*args, **kwargs)

    def openai_error_response(self, *args: Any, **kwargs: Any):
        return self.helpers._openai_error_response(*args, **kwargs)

    def anthropic_error_response(self, *args: Any, **kwargs: Any):
        return self.helpers._anthropic_error_response(*args, **kwargs)

    def native_error_response(self, *args: Any, **kwargs: Any):
        return self.helpers._native_error_response(*args, **kwargs)
