import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

from accio_panel.app_settings import PanelSettings
from accio_panel.config import Settings
from accio_panel.dashboard_views import (
    _filter_accounts_by_status,
    _parse_account_status_filter,
)
from accio_panel.models import Account
from accio_panel.web import create_app


def _account(
    account_id: str,
    *,
    manual_enabled: bool = True,
    auto_disabled: bool = False,
    auto_disabled_reason: str | None = None,
) -> Account:
    return Account(
        id=account_id,
        name=account_id,
        access_token=f"access-{account_id}",
        refresh_token=f"refresh-{account_id}",
        utdid=f"utdid-{account_id}",
        manual_enabled=manual_enabled,
        auto_disabled=auto_disabled,
        auto_disabled_reason=auto_disabled_reason,
    )


class DashboardStatusFilterTests(unittest.TestCase):
    def setUp(self):
        self.accounts = [
            _account("enabled"),
            _account("manual", manual_enabled=False),
            _account("auto", auto_disabled=True, auto_disabled_reason="剩余额度已耗尽"),
            _account(
                "abnormal",
                manual_enabled=False,
                auto_disabled_reason="额度查询失败，且 Token 刷新失败",
            ),
        ]

    def test_parse_account_status_filter_rejects_unknown_values(self):
        self.assertEqual(_parse_account_status_filter(None), "all")
        self.assertEqual(_parse_account_status_filter("enabled"), "enabled")
        self.assertEqual(_parse_account_status_filter("bogus"), "all")

    def test_filter_accounts_by_status(self):
        cases = {
            "all": ["enabled", "manual", "auto", "abnormal"],
            "enabled": ["enabled"],
            "manual_disabled": ["manual"],
            "auto_disabled": ["auto"],
            "abnormal_disabled": ["abnormal"],
        }
        for status_filter, expected_ids in cases.items():
            with self.subTest(status_filter=status_filter):
                filtered = _filter_accounts_by_status(self.accounts, status_filter)
                self.assertEqual([account.id for account in filtered], expected_ids)


class DashboardStatusFilterRouteTests(unittest.TestCase):
    def _build_client(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        async def _noop_scheduler(_application):
            return None

        settings = Settings(data_dir=Path(temp_dir.name), database_url="")
        with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
            app = create_app(settings)

        app.state.panel_settings_store.save(
            PanelSettings(admin_password="admin", session_secret="test-session")
        )
        accounts = [
            _account(f"manual-{index:02d}", manual_enabled=False)
            for index in range(11)
        ]
        accounts.append(
            _account(
                "abnormal",
                manual_enabled=False,
                auto_disabled_reason="额度查询失败，且 Token 刷新失败",
            )
        )
        for account in accounts:
            app.state.store.save(account)

        client = TestClient(app)
        self.addCleanup(client.close)
        response = client.post("/api/auth/login", json={"password": "admin"})
        self.assertEqual(response.status_code, 200)
        return client

    def test_dashboard_status_filter_is_applied_before_pagination(self):
        client = self._build_client()

        response = client.get(
            "/dashboard?view=accounts&status=manual_disabled&page=1&pageSize=10"
        )

        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertIn('id="status-filter-select"', html)
        self.assertIn('value="manual_disabled" selected', html)
        self.assertIn("账号 1 - 10 / 11", html)
        self.assertIn("manual-00", html)
        self.assertNotIn('id="account-name-manual-10"', html)
        self.assertNotIn('id="account-name-abnormal"', html)
        self.assertIn(
            "/dashboard?view=accounts&page=2&pageSize=10&amp;status=manual_disabled",
            html,
        )

    def test_dashboard_empty_filtered_result_keeps_status_filter_visible(self):
        client = self._build_client()

        response = client.get("/dashboard?view=accounts&status=enabled")

        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertIn('id="status-filter-select"', html)
        self.assertIn('value="enabled" selected', html)
        self.assertIn("没有符合条件的账号", html)
        self.assertIn("调整状态过滤条件后再试。", html)


if __name__ == "__main__":
    unittest.main()
