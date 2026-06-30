import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from accio_panel.app_settings import PanelSettings
from accio_panel.models import Account
from accio_panel.proxy_selection import (
    SENTINEL_RATE_LIMIT_INITIAL_SECONDS,
    SENTINEL_RATE_LIMIT_MAX_SECONDS,
    UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON,
    _mark_account_sentinel_rate_limited,
    _clear_account_sentinel_rate_limit,
    _ordered_proxy_candidates_uncached,
    _plan_next_quota_check,
    _query_quota_with_refresh_fallback,
)
from accio_panel.quota_scheduler import _quota_scheduler_loop


class _InMemoryStore:
    def __init__(self, accounts: list[Account]):
        self._accounts = list(accounts)

    def list_accounts(self) -> list[Account]:
        return list(self._accounts)

    def get_account(self, account_id: str) -> Account | None:
        for account in self._accounts:
            if account.id == account_id:
                return account
        return None

    def save(self, account: Account) -> Account:
        for index, existing in enumerate(self._accounts):
            if existing.id == account.id:
                self._accounts[index] = account
                break
        return account

    def update_tokens(
        self,
        account_id: str,
        *,
        access_token: str,
        refresh_token: str,
        expires_at,
    ) -> Account | None:
        account = self.get_account(account_id)
        if not account:
            return None
        account.access_token = access_token
        account.refresh_token = refresh_token
        account.expires_at = expires_at
        self.save(account)
        return account


class _FixedPanelSettingsStore:
    def __init__(self, settings: PanelSettings):
        self._settings = settings

    def load(self) -> PanelSettings:
        return self._settings


class _StopScheduler(Exception):
    pass


class ProxySelectionTests(unittest.TestCase):
    def test_quota_exhausted_recovery_refreshes_token_before_accepting_empty_quota(
        self,
    ):
        account = Account(
            id="acc-1",
            name="账号1",
            access_token="old-access",
            refresh_token="old-refresh",
            utdid="utdid-1",
            auto_disabled=True,
            auto_disabled_reason="上游返回 [429]: quota exhausted",
            next_quota_check_at=1_000_000,
            next_quota_check_reason=UPSTREAM_QUOTA_EXHAUSTED_RECOVERY_REASON,
        )
        store = _InMemoryStore([account])
        client = SimpleNamespace()
        quota_tokens: list[str] = []
        refreshed_tokens: list[str] = []

        def query_quota(account: Account, *, proxy_url=None):
            quota_tokens.append(account.access_token)
            if account.access_token == "old-access":
                return {
                    "success": True,
                    "data": {"remaining": 0, "total": 20},
                }
            return {
                "success": True,
                "data": {"remaining": 20, "total": 20},
            }

        def refresh_token(account: Account, *, proxy_url=None):
            refreshed_tokens.append(account.refresh_token)
            return {
                "success": True,
                "data": {
                    "accessToken": "new-access",
                    "refreshToken": "new-refresh",
                    "expiresAt": 2_000_000,
                },
            }

        client.query_quota = query_quota
        client.refresh_token = refresh_token

        with patch("accio_panel.proxy_selection._now_timestamp", return_value=1_000_000):
            updated, quota = _query_quota_with_refresh_fallback(
                store,
                client,
                account,
                PanelSettings(auto_enable_on_recovered_quota=False),
            )

        self.assertEqual(quota_tokens, ["old-access", "new-access"])
        self.assertEqual(refreshed_tokens, ["old-refresh"])
        self.assertEqual(updated.access_token, "new-access")
        self.assertEqual(updated.refresh_token, "new-refresh")
        self.assertFalse(updated.auto_disabled)
        self.assertIsNone(updated.auto_disabled_reason)
        self.assertEqual(updated.last_remaining_quota, 20)
        self.assertEqual(quota["remaining_text"], "20/20")

    def test_sentinel_rate_limit_cooldown_exponentially_backs_off_and_filters_candidates(self):
        account = Account(
            id="acc-1",
            name="账号1",
            access_token="token-1",
            refresh_token="refresh-1",
            utdid="utdid-1",
        )
        store = _InMemoryStore([account])

        with patch("accio_panel.proxy_selection._now_timestamp", return_value=1_000):
            first = _mark_account_sentinel_rate_limited(store, account)
        self.assertFalse(first.auto_disabled)
        self.assertEqual(
            first.sentinel_rate_limit_backoff_seconds,
            SENTINEL_RATE_LIMIT_INITIAL_SECONDS,
        )
        self.assertEqual(first.sentinel_rate_limited_until, 1_060)

        with patch("accio_panel.proxy_selection._now_timestamp", return_value=1_010):
            second = _mark_account_sentinel_rate_limited(store, first)
        self.assertEqual(second.sentinel_rate_limit_backoff_seconds, 120)
        self.assertEqual(second.sentinel_rate_limited_until, 1_130)

        with patch("accio_panel.proxy_selection._now_timestamp", return_value=1_020):
            third = _mark_account_sentinel_rate_limited(store, second)

        self.assertFalse(third.auto_disabled)
        self.assertEqual(third.sentinel_rate_limit_backoff_seconds, 240)
        self.assertEqual(third.sentinel_rate_limited_until, 1_260)

        with patch("accio_panel.proxy_selection._now_timestamp", return_value=1_100):
            self.assertEqual(
                _ordered_proxy_candidates_uncached(store, None, None, None),
                [],
            )
        with patch("accio_panel.proxy_selection._now_timestamp", return_value=1_261):
            self.assertEqual(
                [
                    candidate.id
                    for candidate in _ordered_proxy_candidates_uncached(
                        store,
                        None,
                        None,
                        None,
                    )
                ],
                ["acc-1"],
            )

        for index in range(10):
            with patch(
                "accio_panel.proxy_selection._now_timestamp",
                return_value=2_000 + index,
            ):
                account = _mark_account_sentinel_rate_limited(store, account)
        self.assertEqual(
            account.sentinel_rate_limit_backoff_seconds,
            SENTINEL_RATE_LIMIT_MAX_SECONDS,
        )

    def test_sentinel_rate_limit_cooldown_clears_after_success(self):
        account = Account(
            id="acc-1",
            name="账号1",
            access_token="token-1",
            refresh_token="refresh-1",
            utdid="utdid-1",
            sentinel_rate_limited_until=1_060,
            sentinel_rate_limit_backoff_seconds=240,
        )
        store = _InMemoryStore([account])

        with patch("accio_panel.proxy_selection._now_timestamp", return_value=2_000):
            cleared = _clear_account_sentinel_rate_limit(store, account)
        self.assertIsNone(cleared.sentinel_rate_limited_until)
        self.assertEqual(cleared.sentinel_rate_limit_backoff_seconds, 0)

        with patch("accio_panel.proxy_selection._now_timestamp", return_value=2_010):
            limited = _mark_account_sentinel_rate_limited(store, cleared)
        self.assertEqual(
            limited.sentinel_rate_limit_backoff_seconds,
            SENTINEL_RATE_LIMIT_INITIAL_SECONDS,
        )
        self.assertEqual(limited.sentinel_rate_limited_until, 2_070)

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
    async def test_scheduler_limits_due_account_checks_to_startup_batch_size(self):
        accounts = [
            Account(
                id=f"acc-{index}",
                name=f"账号{index}",
                access_token=f"access-{index}",
                refresh_token=f"refresh-{index}",
                utdid=f"utdid-{index}",
                manual_enabled=True,
                next_quota_check_at=None,
            )
            for index in range(25)
        ]
        application = SimpleNamespace(
            state=SimpleNamespace(
                store=_InMemoryStore(accounts),
                client=object(),
                panel_settings_store=_FixedPanelSettingsStore(PanelSettings()),
            )
        )
        calls: list[str] = []

        def fake_query_quota_with_refresh_fallback(_store, _client, account, _settings):
            calls.append(account.id)
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

        self.assertEqual(calls, [f"acc-{index}" for index in range(10)])

    async def test_scheduler_sleeps_until_next_startup_stagger_batch(self):
        accounts = [
            Account(
                id=f"acc-{index}",
                name=f"账号{index}",
                access_token=f"access-{index}",
                refresh_token=f"refresh-{index}",
                utdid=f"utdid-{index}",
                manual_enabled=True,
                next_quota_check_at=1_003_600,
            )
            for index in range(15)
        ]
        store = _InMemoryStore(accounts)
        application = SimpleNamespace(
            state=SimpleNamespace(
                store=store,
                client=object(),
                panel_settings_store=_FixedPanelSettingsStore(PanelSettings()),
            )
        )
        calls: list[str] = []
        sleeps: list[float] = []

        def fake_query_quota_with_refresh_fallback(_store, _client, account, _settings):
            calls.append(account.id)
            account.next_quota_check_at = 1_000_900
            store.save(account)
            return account, {"success": True, "message": ""}

        async def stop_after_first_sleep(delay: float):
            sleeps.append(delay)
            raise _StopScheduler()

        with (
            patch("accio_panel.quota_scheduler._now_timestamp", return_value=1_000_000),
            patch(
                "accio_panel.quota_scheduler._query_quota_with_refresh_fallback",
                side_effect=fake_query_quota_with_refresh_fallback,
            ),
            patch("accio_panel.quota_scheduler.asyncio.sleep", side_effect=stop_after_first_sleep),
        ):
            with self.assertRaises(_StopScheduler):
                await _quota_scheduler_loop(application)

        self.assertEqual(calls, [f"acc-{index}" for index in range(10)])
        self.assertEqual(sleeps, [2])

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
