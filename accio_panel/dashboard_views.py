from __future__ import annotations

from typing import Any

from .models import Account
from .proxy_selection import _account_status_view
from .utils import format_timestamp, mask_token

PAGE_SIZE_OPTIONS = (10, 20, 50)
DEFAULT_PAGE_SIZE = PAGE_SIZE_OPTIONS[0]
ACCOUNT_STATUS_FILTER_OPTIONS = (
    {"value": "all", "label": "全部状态"},
    {"value": "enabled", "label": "已启用"},
    {"value": "manual_disabled", "label": "手动禁用"},
    {"value": "auto_disabled", "label": "自动禁用"},
    {"value": "abnormal_disabled", "label": "异常禁用"},
)
ACCOUNT_STATUS_FILTER_VALUES = frozenset(
    item["value"] for item in ACCOUNT_STATUS_FILTER_OPTIONS
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


def _parse_account_status_filter(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in ACCOUNT_STATUS_FILTER_VALUES else "all"


def _filter_accounts_by_status(
    accounts: list[Account],
    status_filter: str,
) -> list[Account]:
    if status_filter == "enabled":
        return [
            account
            for account in accounts
            if account.manual_enabled and not account.auto_disabled
        ]
    if status_filter == "manual_disabled":
        return [
            account
            for account in accounts
            if not account.manual_enabled
            and not str(account.auto_disabled_reason or "").strip()
        ]
    if status_filter == "auto_disabled":
        return [
            account
            for account in accounts
            if account.manual_enabled and account.auto_disabled
        ]
    if status_filter == "abnormal_disabled":
        return [
            account
            for account in accounts
            if not account.manual_enabled
            and str(account.auto_disabled_reason or "").strip()
        ]
    return list(accounts)


def _build_page_numbers(current_page: int, total_pages: int) -> list[int]:
    if total_pages <= 7:
        return list(range(1, total_pages + 1))
    start = max(1, current_page - 2)
    end = min(total_pages, start + 4)
    start = max(1, end - 4)
    return list(range(start, end + 1))


def _cached_quota_view(account: Account) -> dict[str, Any]:
    remaining = account.last_remaining_quota
    total = account.last_total_quota

    if remaining is None:
        checked = account.last_quota_check_at is not None
        return {
            "success": False,
            "total_value": 0,
            "used_value": 0,
            "used_text": "-",
            "remaining_value": 0,
            "remaining_ratio": 0,
            "remaining_text": "获取失败" if checked else "等待巡检",
            "reset_text": "-",
            "level": "error",
            "message": account.auto_disabled_reason or "",
        }

    remaining = max(0, remaining)
    total = max(0, total or 0)
    used = max(0, total - remaining) if total > 0 else 0
    if total <= 0 and (used > 0 or remaining > 0):
        total = used + remaining
    ratio = max(0, min(100, round((remaining / total) * 100))) if total > 0 else 0

    level = "low"
    if ratio < 20:
        level = "high"
    elif ratio < 50:
        level = "medium"

    fmt = lambda v, t: f"{v}/{t}" if t > 0 else str(v)
    return {
        "success": True,
        "total_value": total,
        "used_value": used,
        "used_text": fmt(used, total),
        "remaining_value": remaining,
        "remaining_ratio": ratio,
        "remaining_text": fmt(remaining, total),
        "reset_text": "-",
        "level": level,
        "message": "",
    }


def _build_dashboard_items(
    accounts: list[Account],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for account in accounts:
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
                "quota": _cached_quota_view(account),
            }
        )
    return items
