from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI

from .app_settings import PanelSettings
from .client import AccioClient
from .gemini_proxy import (
    build_gemini_model_payload,
    normalize_gemini_model_name,
)
from .model_catalog import (
    build_gemini_model_payload_from_catalog,
    extract_model_catalog,
    is_image_generation_model,
    list_model_names,
    list_proxy_model_names,
)
from .proxy_selection import (
    _query_llm_config_with_refresh_fallback,
    _sorted_enabled_accounts,
)
from .store import AccountStore

MODEL_CATALOG_CACHE_SECONDS = 60


def _initial_model_catalog_cache() -> dict[str, Any]:
    return {
        "entries": [],
        "expiresAt": 0.0,
        "loadedAt": 0.0,
        "sourceAccountId": "",
        "error": "",
    }


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
