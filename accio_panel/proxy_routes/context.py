from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from ..api_logs import ApiLogStore
from ..app_settings import PanelSettingsStore
from ..client import AccioClient
from ..config import Settings
from ..gemini_proxy import (
    decode_gemini_generate_content_response as _decode_gemini_generate_content_response,
)
from ..model_catalog_cache import (
    _is_allowed_dynamic_model as _is_allowed_dynamic_model_impl,
)
from ..proxy_selection import (
    ProxySelectionError as _ProxySelectionError,
    _anthropic_error_response as _anthropic_error_response_impl,
    _authorize_proxy_request as _authorize_proxy_request_impl,
    _disable_account_model_on_empty_response as _disable_account_model_on_empty_response_impl,
    _empty_response_log_message as _empty_response_log_message_impl,
    _extract_proxy_api_key as _extract_proxy_api_key_impl,
    _gemini_error_response as _gemini_error_response_impl,
    _iter_upstream_sse_bytes as _iter_upstream_sse_bytes_impl,
    _mark_account_quota_exhausted_cooldown as _mark_account_quota_exhausted_cooldown_impl,
    _native_error_response as _native_error_response_impl,
    _openai_error_response as _openai_error_response_impl,
    _select_proxy_account as _select_proxy_account_impl,
    _should_disable_model_on_empty_response as _should_disable_model_on_empty_response_impl,
)
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
    def ProxySelectionError(self):
        return _ProxySelectionError

    def authorize_proxy_request(self, *args: Any, **kwargs: Any):
        return _authorize_proxy_request_impl(*args, **kwargs)

    def is_allowed_dynamic_model(self, *args: Any, **kwargs: Any):
        return _is_allowed_dynamic_model_impl(*args, **kwargs)

    def select_proxy_account(self, *args: Any, **kwargs: Any):
        return _select_proxy_account_impl(*args, **kwargs)

    def empty_response_log_message(self, *args: Any, **kwargs: Any):
        return _empty_response_log_message_impl(*args, **kwargs)

    def should_disable_model_on_empty_response(self, *args: Any, **kwargs: Any):
        return _should_disable_model_on_empty_response_impl(*args, **kwargs)

    def disable_account_model_on_empty_response(self, *args: Any, **kwargs: Any):
        return _disable_account_model_on_empty_response_impl(*args, **kwargs)

    def mark_account_quota_exhausted_cooldown(self, *args: Any, **kwargs: Any):
        return _mark_account_quota_exhausted_cooldown_impl(*args, **kwargs)

    def extract_proxy_api_key(self, *args: Any, **kwargs: Any):
        return _extract_proxy_api_key_impl(*args, **kwargs)

    def iter_upstream_sse_bytes(self, *args: Any, **kwargs: Any):
        return _iter_upstream_sse_bytes_impl(*args, **kwargs)

    def gemini_error_response(self, *args: Any, **kwargs: Any):
        return _gemini_error_response_impl(*args, **kwargs)

    def decode_gemini_generate_content_response(self, *args: Any, **kwargs: Any):
        return _decode_gemini_generate_content_response(*args, **kwargs)

    def openai_error_response(self, *args: Any, **kwargs: Any):
        return _openai_error_response_impl(*args, **kwargs)

    def anthropic_error_response(self, *args: Any, **kwargs: Any):
        return _anthropic_error_response_impl(*args, **kwargs)

    def native_error_response(self, *args: Any, **kwargs: Any):
        return _native_error_response_impl(*args, **kwargs)
