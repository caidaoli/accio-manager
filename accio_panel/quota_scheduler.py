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


async def _quota_scheduler_loop(application: FastAPI) -> None:
    store: AccountStore = application.state.store
    client: AccioClient = application.state.client
    panel_settings_store: PanelSettingsStore = application.state.panel_settings_store

    # 启动时分批重置账号巡检时间，避免惊群效应
    # 策略：每批 10 个账号，间隔 2 秒，分散上游 API 压力
    now_ts = _now_timestamp()
    accounts = store.list_accounts()

    for i, account in enumerate(accounts):
        if account.next_quota_check_at is not None and account.next_quota_check_at > now_ts:
            # 计算分散时间：第 0-9 个账号在 now，第 10-19 个在 now+2s，依此类推
            batch_index = i // QUOTA_CHECK_BATCH_SIZE
            stagger_offset = batch_index * QUOTA_CHECK_BATCH_INTERVAL_SECONDS
            account.next_quota_check_at = now_ts + stagger_offset
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

        await asyncio.sleep(SCHEDULER_TICK_SECONDS)
