from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .app_settings import PanelSettings
from .client import AccioClient
from .models import Account
from .proxy_selection import (
    _account_status_view,
    _build_quota_view,
    _query_quota_with_refresh_fallback,
)
from .store import AccountStore
from .utils import format_timestamp, mask_token

PAGE_SIZE_OPTIONS = (10, 20, 50)
DEFAULT_PAGE_SIZE = PAGE_SIZE_OPTIONS[0]


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
