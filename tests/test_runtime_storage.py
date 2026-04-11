import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from accio_panel.app_settings import PanelSettings, PanelSettingsStore
from accio_panel.config import Settings
from accio_panel.models import Account
from accio_panel.persistence import create_runtime_stores
from accio_panel.store import AccountStore


class _FakeGateway:
    def __init__(self):
        self.panel_settings: dict[str, object] | None = None
        self.accounts: dict[str, dict[str, object]] = {}
        self.ensure_schema_calls = 0

    def ensure_schema(self) -> None:
        self.ensure_schema_calls += 1

    def fetch_panel_settings(self) -> dict[str, object] | None:
        if self.panel_settings is None:
            return None
        return dict(self.panel_settings)

    def save_panel_settings(self, payload: dict[str, object]) -> None:
        self.panel_settings = dict(payload)

    def count_accounts(self) -> int:
        return len(self.accounts)

    def list_accounts(self) -> list[dict[str, object]]:
        return [
            dict(item)
            for item in sorted(
                self.accounts.values(),
                key=lambda item: (str(item.get("addedAt") or ""), str(item.get("name") or ""), str(item.get("id") or "")),
            )
        ]

    def upsert_account(self, payload: dict[str, object]) -> None:
        self.accounts[str(payload["id"])] = dict(payload)

    def delete_account(self, account_id: str) -> bool:
        return self.accounts.pop(account_id, None) is not None


class RuntimeStorageTests(unittest.TestCase):
    def test_settings_version_is_fixed_and_ignores_accio_version_env(self):
        import accio_panel.config as config_module

        with patch.dict(
            os.environ,
            {"ACCIO_VERSION": "9.9.9"},
            clear=False,
        ):
            reloaded = importlib.reload(config_module)
            try:
                settings = reloaded.Settings()
                self.assertEqual(settings.version, "0.8.8")
            finally:
                importlib.reload(config_module)

    def test_settings_uses_binary_adjacent_data_dir_when_running_from_nuitka(self):
        import accio_panel.config as config_module

        compiled = type("Compiled", (), {"containing_dir": "/tmp/accio-dist"})()

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(config_module, "__compiled__", compiled, create=True):
                settings = config_module.Settings()

        self.assertEqual(settings.data_dir, Path("/tmp/accio-dist/data"))

    def test_settings_prefers_accio_data_dir_over_compiled_default(self):
        import accio_panel.config as config_module

        compiled = type("Compiled", (), {"containing_dir": "/tmp/accio-dist"})()

        with patch.dict(
            os.environ,
            {"ACCIO_DATA_DIR": "/var/lib/accio-data"},
            clear=True,
        ):
            with patch.object(config_module, "__compiled__", compiled, create=True):
                settings = config_module.Settings()

        self.assertEqual(settings.data_dir, Path("/var/lib/accio-data"))

    def test_settings_reads_mysql_dsn_from_accio_mysql(self):
        import accio_panel.config as config_module

        with patch.dict(
            os.environ,
            {
                "ACCIO_MYSQL": "mysql://root:secret@127.0.0.1:3306/accio",
                "ACCIO_DATABASE_URL": "",
            },
            clear=False,
        ):
            reloaded = importlib.reload(config_module)
            try:
                settings = reloaded.Settings()
                self.assertEqual(
                    settings.database_url,
                    "mysql://root:secret@127.0.0.1:3306/accio",
                )
                self.assertTrue(settings.database_enabled)
            finally:
                importlib.reload(config_module)

    def test_create_runtime_stores_returns_file_stores_without_database_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                data_dir=Path(temp_dir),
                database_url="",
            )

            account_store, panel_settings_store = create_runtime_stores(settings)

            self.assertIsInstance(account_store, AccountStore)
            self.assertIsInstance(panel_settings_store, PanelSettingsStore)

    def test_create_runtime_stores_bootstraps_database_when_tables_are_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                data_dir=Path(temp_dir),
                database_url="mysql://root:secret@127.0.0.1:3306/accio",
            )

            file_account_store = AccountStore(settings.accounts_dir, settings.accounts_file)
            file_account_store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="access-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    added_at="2026-04-04 00:00:00",
                    updated_at="2026-04-04 00:00:00",
                )
            )

            file_panel_store = PanelSettingsStore(
                settings.settings_file,
                settings.legacy_settings_file,
            )
            file_panel_store.save(
                PanelSettings(
                    upstream_proxy_url="http://127.0.0.1:7890",
                    auto_disable_on_empty_quota=False,
                    auto_enable_on_recovered_quota=True,
                    api_account_strategy="round_robin",
                    admin_password="db-admin",
                    session_secret="session-from-file",
                )
            )

            gateway = _FakeGateway()
            with patch("accio_panel.persistence.build_mysql_gateway", return_value=gateway):
                account_store, panel_settings_store = create_runtime_stores(settings)

            loaded_accounts = account_store.list_accounts()
            loaded_settings = panel_settings_store.load()

            self.assertEqual(len(loaded_accounts), 1)
            self.assertEqual(loaded_accounts[0].name, "账号1")
            self.assertEqual(loaded_settings.admin_password, "db-admin")
            self.assertEqual(loaded_settings.api_account_strategy, "round_robin")

    def test_create_runtime_stores_prefers_database_data_when_database_has_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                data_dir=Path(temp_dir),
                database_url="mysql://root:secret@127.0.0.1:3306/accio",
            )

            file_account_store = AccountStore(settings.accounts_dir, settings.accounts_file)
            file_account_store.save(
                Account(
                    id="local-acc",
                    name="本地账号",
                    access_token="local-access",
                    refresh_token="local-refresh",
                    utdid="local-utdid",
                    added_at="2026-04-04 00:00:00",
                    updated_at="2026-04-04 00:00:00",
                )
            )

            file_panel_store = PanelSettingsStore(
                settings.settings_file,
                settings.legacy_settings_file,
            )
            file_panel_store.save(
                PanelSettings(
                    admin_password="local-admin",
                    session_secret="local-session",
                )
            )

            gateway = _FakeGateway()
            gateway.panel_settings = {
                "upstreamProxyUrl": "",
                "autoDisableOnEmptyQuota": True,
                "autoEnableOnRecoveredQuota": True,
                "apiAccountStrategy": "fill",
                "adminPassword": "mysql-admin",
                "sessionSecret": "mysql-session",
            }
            gateway.accounts["db-acc"] = {
                "id": "db-acc",
                "name": "数据库账号",
                "accessToken": "db-access",
                "refreshToken": "db-refresh",
                "utdid": "db-utdid",
                "fillPriority": 5,
                "expiresAt": None,
                "cookie": None,
                "manualEnabled": True,
                "autoDisabled": False,
                "autoDisabledReason": None,
                "lastQuotaCheckAt": None,
                "nextQuotaCheckAt": None,
                "nextQuotaCheckReason": None,
                "addedAt": "2026-04-04 01:00:00",
                "updatedAt": "2026-04-04 01:00:00",
            }

            with patch("accio_panel.persistence.build_mysql_gateway", return_value=gateway):
                account_store, panel_settings_store = create_runtime_stores(settings)

            loaded_accounts = account_store.list_accounts()
            loaded_settings = panel_settings_store.load()

            self.assertEqual(len(loaded_accounts), 1)
            self.assertEqual(loaded_accounts[0].id, "db-acc")
            self.assertEqual(loaded_accounts[0].name, "数据库账号")
            self.assertEqual(loaded_settings.admin_password, "mysql-admin")


if __name__ == "__main__":
    unittest.main()
