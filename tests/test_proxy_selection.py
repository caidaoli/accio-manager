import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from accio_panel.app_settings import PanelSettings
from accio_panel.models import Account
from accio_panel.proxy_selection import (
    UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON,
    _plan_next_quota_check,
)
from accio_panel.quota_scheduler import _quota_scheduler_loop


class _InMemoryStore:
    def __init__(self, accounts: list[Account]):
        self._accounts = list(accounts)

    def list_accounts(self) -> list[Account]:
        return list(self._accounts)

    def save(self, account: Account) -> Account:
        for index, existing in enumerate(self._accounts):
            if existing.id == account.id:
                self._accounts[index] = account
                break
        return account


class _FixedPanelSettingsStore:
    def __init__(self, settings: PanelSettings):
        self._settings = settings

    def load(self) -> PanelSettings:
        return self._settings


class _StopScheduler(Exception):
    pass


class ProxySelectionTests(unittest.TestCase):
    def test_quota_exhausted_recovery_is_capped_when_next_billing_is_far_away(self):
        account = Account(
            id="acc-1",
            name="账号1",
            access_token="access-1",
            refresh_token="refresh-1",
            utdid="utdid-1",
            auto_disabled=True,
            next_quota_check_reason=UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON,
        )

        next_check_at, reason = _plan_next_quota_check(
            account,
            quota_success=True,
            next_billing_at=1_800_000,
            panel_settings=PanelSettings(),
            now_ts=1_000_000,
        )

        self.assertEqual(next_check_at, 1_001_800)
        self.assertEqual(reason, UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON)

    def test_quota_exhausted_recovery_keeps_near_billing_retry_time(self):
        account = Account(
            id="acc-1",
            name="账号1",
            access_token="access-1",
            refresh_token="refresh-1",
            utdid="utdid-1",
            auto_disabled=True,
            next_quota_check_reason=UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON,
        )

        next_check_at, reason = _plan_next_quota_check(
            account,
            quota_success=True,
            next_billing_at=1_000_300,
            panel_settings=PanelSettings(),
            now_ts=1_000_000,
        )

        self.assertEqual(next_check_at, 1_000_390)
        self.assertEqual(reason, UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON)


class QuotaSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_scheduler_recovery_uses_same_quota_refresh_path_for_abnormal_accounts(
        self,
    ):
        account = Account(
            id="acc-1",
            name="账号1",
            access_token="access-1",
            refresh_token="refresh-1",
            utdid="utdid-1",
            manual_enabled=False,
            auto_disabled_reason="额度查询失败，且 Token 刷新失败",
            next_quota_check_at=1_000_000,
            next_quota_check_reason="异常禁用后定时恢复检查",
        )
        application = SimpleNamespace(
            state=SimpleNamespace(
                store=_InMemoryStore([account]),
                client=object(),
                panel_settings_store=_FixedPanelSettingsStore(PanelSettings()),
            )
        )
        calls: list[str] = []

        def fake_query_quota_with_refresh_fallback(*args, **kwargs):
            calls.append("quota")
            return account, {"success": False, "message": "额度查询失败"}

        async def stop_after_first_tick(_: float):
            raise _StopScheduler()

        with (
            patch("accio_panel.quota_scheduler._now_timestamp", return_value=1_000_000),
            patch(
                "accio_panel.quota_scheduler._query_quota_with_refresh_fallback",
                side_effect=fake_query_quota_with_refresh_fallback,
            ),
            patch("accio_panel.quota_scheduler.asyncio.sleep", side_effect=stop_after_first_tick),
        ):
            with self.assertRaises(_StopScheduler):
                await _quota_scheduler_loop(application)

        self.assertEqual(calls, ["quota"])

    async def test_scheduler_skips_manual_disabled_accounts_without_abnormal_reason(self):
        account = Account(
            id="acc-1",
            name="账号1",
            access_token="access-1",
            refresh_token="refresh-1",
            utdid="utdid-1",
            manual_enabled=False,
            next_quota_check_at=1_000_000,
            next_quota_check_reason="手动切换启用状态后立即检查额度",
        )
        store = _InMemoryStore([account])
        application = SimpleNamespace(
            state=SimpleNamespace(
                store=store,
                client=object(),
                panel_settings_store=_FixedPanelSettingsStore(PanelSettings()),
            )
        )
        calls: list[str] = []

        def fake_query_quota_with_refresh_fallback(*args, **kwargs):
            calls.append("quota")
            return account, {"success": True, "message": ""}

        async def stop_after_first_tick(_: float):
            raise _StopScheduler()

        with (
            patch("accio_panel.quota_scheduler._now_timestamp", return_value=1_000_000),
            patch(
                "accio_panel.quota_scheduler._query_quota_with_refresh_fallback",
                side_effect=fake_query_quota_with_refresh_fallback,
            ),
            patch("accio_panel.quota_scheduler.asyncio.sleep", side_effect=stop_after_first_tick),
        ):
            with self.assertRaises(_StopScheduler):
                await _quota_scheduler_loop(application)

        self.assertEqual(calls, [])
        self.assertIsNone(account.next_quota_check_at)
        self.assertIsNone(account.next_quota_check_reason)


if __name__ == "__main__":
    unittest.main()
