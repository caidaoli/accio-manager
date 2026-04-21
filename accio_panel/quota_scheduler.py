from __future__ import annotations

import asyncio

from fastapi import FastAPI

from .app_settings import PanelSettingsStore
from .client import AccioClient
from .models import Account
from .proxy_selection import (
    _now_timestamp,
    _query_quota_with_refresh_fallback,
)
from .store import AccountStore

SCHEDULER_TICK_SECONDS = 30


async def _quota_scheduler_loop(application: FastAPI) -> None:
    store: AccountStore = application.state.store
    client: AccioClient = application.state.client
    panel_settings_store: PanelSettingsStore = application.state.panel_settings_store

    # 启动时重置所有账号的巡检时间，确保首轮 tick 立即触发全量检查
    now_ts = _now_timestamp()
    for account in store.list_accounts():
        if account.next_quota_check_at is not None and account.next_quota_check_at > now_ts:
            account.next_quota_check_at = now_ts
            store.save(account)

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
                _query_quota_with_refresh_fallback,
                store,
                client,
                account,
                panel_settings,
            )

        await asyncio.sleep(SCHEDULER_TICK_SECONDS)
