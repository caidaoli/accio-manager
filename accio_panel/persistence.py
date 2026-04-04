from __future__ import annotations

from .app_settings import PanelSettingsStore
from .config import Settings
from .mysql_storage import MySQLAccountStore, MySQLGateway, MySQLPanelSettingsStore
from .store import AccountStore


def build_mysql_gateway(settings: Settings) -> MySQLGateway:
    return MySQLGateway.from_settings(settings)


def create_runtime_stores(settings: Settings) -> tuple[object, object]:
    file_account_store = AccountStore(settings.accounts_dir, settings.accounts_file)
    file_panel_settings_store = PanelSettingsStore(
        settings.settings_file,
        settings.legacy_settings_file,
    )
    if not settings.database_enabled:
        return file_account_store, file_panel_settings_store

    gateway = build_mysql_gateway(settings)
    panel_settings_store = MySQLPanelSettingsStore(gateway)
    account_store = MySQLAccountStore(gateway)
    panel_settings_store.bootstrap_from_file_if_empty(file_panel_settings_store)
    account_store.bootstrap_from_file_if_empty(file_account_store)
    return account_store, panel_settings_store
