from __future__ import annotations

import asyncio

from fastapi import FastAPI

from .app_settings import PanelSettingsStore
from .client import AccioClient
from .models import Account
from .proxy_selection import (
    QUOTA_SCHEDULER_TICK_SECONDS,
    _now_timestamp,
    _query_quota_with_refresh_fallback,
)
from .store import AccountStore

SCHEDULER_TICK_SECONDS = QUOTA_SCHEDULER_TICK_SECONDS
QUOTA_CHECK_BATCH_SIZE = 10
QUOTA_CHECK_BATCH_INTERVAL_SECONDS = 2
SCHEDULED_QUOTA_CHECK_REASONS = {
    "额度查询失败后重试",
    "额度查询失败后自动刷新 Token 并重试额度",
    "上游 quota exhausted 后自动恢复重试",
    "自动恢复重试",
    "等待额度重置后自动恢复检查",
}


async def _run_quota_check_batch(
    store: AccountStore,
    client: AccioClient,
    accounts: list[Account],
    panel_settings,
) -> None:
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
        for account in accounts
    ]
    await asyncio.gather(*tasks, return_exceptions=True)


def _has_scheduled_quota_check(account: Account) -> bool:
    return (
        account.manual_enabled
        and account.next_quota_check_at is not None
        and str(account.next_quota_check_reason or "").strip()
        in SCHEDULED_QUOTA_CHECK_REASONS
    )


def _has_due_scheduled_quota_check(account: Account, now_ts: int) -> bool:
    return _has_scheduled_quota_check(account) and account.next_quota_check_at <= now_ts


def _has_pending_abnormal_recovery_check(account: Account) -> bool:
    return (
        not account.manual_enabled
        and bool(str(account.auto_disabled_reason or "").strip())
        and account.next_quota_check_at is not None
    )


def _next_scheduler_sleep_seconds(store: AccountStore) -> int:
    now_ts = _now_timestamp()
    next_check_times = [
        int(account.next_quota_check_at)
        for account in store.list_accounts()
        if _has_scheduled_quota_check(account)
        or _has_pending_abnormal_recovery_check(account)
    ]
    if not next_check_times:
        return SCHEDULER_TICK_SECONDS
    return min(
        SCHEDULER_TICK_SECONDS,
        max(0, min(next_check_times) - now_ts),
    )


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
                if _has_pending_abnormal_recovery_check(account) and (
                    account.next_quota_check_at <= now_ts
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

            if _has_due_scheduled_quota_check(account, now_ts):
                due_accounts.append(account)

        if due_accounts:
            for start in range(0, len(due_accounts), QUOTA_CHECK_BATCH_SIZE):
                batch = due_accounts[start : start + QUOTA_CHECK_BATCH_SIZE]
                await _run_quota_check_batch(
                    store,
                    client,
                    batch,
                    panel_settings,
                )
                if start + QUOTA_CHECK_BATCH_SIZE < len(due_accounts):
                    await asyncio.sleep(QUOTA_CHECK_BATCH_INTERVAL_SECONDS)

        for account in abnormal_recovery_accounts:
            await asyncio.to_thread(
                _query_quota_with_refresh_fallback,
                store,
                client,
                account,
                panel_settings,
            )

        await asyncio.sleep(_next_scheduler_sleep_seconds(store))
