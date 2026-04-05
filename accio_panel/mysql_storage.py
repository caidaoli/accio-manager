from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .app_settings import (
    PanelSettings,
    load_panel_settings,
    normalize_panel_settings,
)
from .config import Settings
from .models import Account
from .store import BaseAccountStore


_MYSQL_SCHEMES = {"mysql", "mysql+pymysql"}


class MySQLGateway:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        charset: str = "utf8mb4",
        use_ssl: bool = False,
        ssl_ca: str = "",
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset or "utf8mb4"
        self.use_ssl = use_ssl
        self.ssl_ca = ssl_ca
        self._conn = None
        self._conn_lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "MySQLGateway":
        if not settings.database_url:
            raise ValueError("未配置 ACCIO_MYSQL")
        return cls(**_parse_database_url(settings.database_url))

    def _new_connection(self):
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError(
                "数据库模式需要安装 PyMySQL，请先执行 `uv sync` 安装依赖。"
            ) from exc

        kwargs: dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": self.charset,
            "autocommit": True,
            "cursorclass": pymysql.cursors.DictCursor,
        }
        if self.use_ssl:
            if self.ssl_ca:
                kwargs["ssl_ca"] = self.ssl_ca
                kwargs["ssl_verify_cert"] = True
            else:
                kwargs["ssl"] = {"verify_mode": False}
        return pymysql.connect(**kwargs)

    def _get_conn(self):
        with self._conn_lock:
            conn = self._conn
            if conn is None:
                conn = self._new_connection()
                self._conn = conn
            else:
                try:
                    conn.ping(reconnect=True)
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = self._new_connection()
                    self._conn = conn
            return conn

    def _discard_conn(self, conn):
        with self._conn_lock:
            if self._conn is conn:
                self._conn = None
        try:
            conn.close()
        except Exception:
            pass

    @contextmanager
    def _connect(self):
        conn = self._get_conn()
        try:
            yield conn
        except Exception:
            self._discard_conn(conn)
            raise

    def ensure_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS accio_panel_settings (
                        id TINYINT PRIMARY KEY,
                        upstream_proxy_url TEXT NOT NULL,
                        auto_disable_on_empty_quota BOOLEAN NOT NULL,
                        auto_enable_on_recovered_quota BOOLEAN NOT NULL,
                        api_account_strategy VARCHAR(32) NOT NULL,
                        admin_password VARCHAR(255) NOT NULL,
                        session_secret VARCHAR(255) NOT NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP
                    ) DEFAULT CHARSET=utf8mb4
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS accio_accounts (
                        id VARCHAR(64) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        access_token TEXT NOT NULL,
                        refresh_token TEXT NOT NULL,
                        utdid VARCHAR(255) NOT NULL,
                        fill_priority INT NOT NULL DEFAULT 100,
                        expires_at BIGINT NULL,
                        cookie LONGTEXT NULL,
                        manual_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                        auto_disabled BOOLEAN NOT NULL DEFAULT FALSE,
                        auto_disabled_reason TEXT NULL,
                        last_quota_check_at BIGINT NULL,
                        next_quota_check_at BIGINT NULL,
                        next_quota_check_reason TEXT NULL,
                        added_at VARCHAR(19) NOT NULL,
                        updated_at VARCHAR(19) NOT NULL
                    ) DEFAULT CHARSET=utf8mb4
                    """
                )

    def fetch_panel_settings(self) -> dict[str, object] | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        upstream_proxy_url,
                        auto_disable_on_empty_quota,
                        auto_enable_on_recovered_quota,
                        api_account_strategy,
                        admin_password,
                        session_secret
                    FROM accio_panel_settings
                    WHERE id = 1
                    """
                )
                row = cursor.fetchone()

        if not row:
            return None

        return {
            "upstreamProxyUrl": row.get("upstream_proxy_url") or "",
            "autoDisableOnEmptyQuota": bool(
                row.get("auto_disable_on_empty_quota", True)
            ),
            "autoEnableOnRecoveredQuota": bool(
                row.get("auto_enable_on_recovered_quota", True)
            ),
            "apiAccountStrategy": row.get("api_account_strategy") or "fill",
            "adminPassword": row.get("admin_password") or "",
            "sessionSecret": row.get("session_secret") or "",
        }

    def save_panel_settings(self, payload: dict[str, object]) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO accio_panel_settings (
                        id,
                        upstream_proxy_url,
                        auto_disable_on_empty_quota,
                        auto_enable_on_recovered_quota,
                        api_account_strategy,
                        admin_password,
                        session_secret
                    ) VALUES (1, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        upstream_proxy_url = VALUES(upstream_proxy_url),
                        auto_disable_on_empty_quota = VALUES(auto_disable_on_empty_quota),
                        auto_enable_on_recovered_quota = VALUES(auto_enable_on_recovered_quota),
                        api_account_strategy = VALUES(api_account_strategy),
                        admin_password = VALUES(admin_password),
                        session_secret = VALUES(session_secret)
                    """,
                    (
                        str(payload.get("upstreamProxyUrl") or ""),
                        bool(payload.get("autoDisableOnEmptyQuota", True)),
                        bool(payload.get("autoEnableOnRecoveredQuota", True)),
                        str(payload.get("apiAccountStrategy") or "fill"),
                        str(payload.get("adminPassword") or ""),
                        str(payload.get("sessionSecret") or ""),
                    ),
                )

    def count_accounts(self) -> int:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) AS total FROM accio_accounts")
                row = cursor.fetchone() or {}
        return int(row.get("total") or 0)

    def list_accounts(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        id,
                        name,
                        access_token,
                        refresh_token,
                        utdid,
                        fill_priority,
                        expires_at,
                        cookie,
                        manual_enabled,
                        auto_disabled,
                        auto_disabled_reason,
                        last_quota_check_at,
                        next_quota_check_at,
                        next_quota_check_reason,
                        added_at,
                        updated_at
                    FROM accio_accounts
                    ORDER BY added_at, name, id
                    """
                )
                rows = cursor.fetchall() or []

        return [_account_row_to_payload(row) for row in rows]

    def upsert_account(self, payload: dict[str, object]) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO accio_accounts (
                        id,
                        name,
                        access_token,
                        refresh_token,
                        utdid,
                        fill_priority,
                        expires_at,
                        cookie,
                        manual_enabled,
                        auto_disabled,
                        auto_disabled_reason,
                        last_quota_check_at,
                        next_quota_check_at,
                        next_quota_check_reason,
                        added_at,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        name = VALUES(name),
                        access_token = VALUES(access_token),
                        refresh_token = VALUES(refresh_token),
                        utdid = VALUES(utdid),
                        fill_priority = VALUES(fill_priority),
                        expires_at = VALUES(expires_at),
                        cookie = VALUES(cookie),
                        manual_enabled = VALUES(manual_enabled),
                        auto_disabled = VALUES(auto_disabled),
                        auto_disabled_reason = VALUES(auto_disabled_reason),
                        last_quota_check_at = VALUES(last_quota_check_at),
                        next_quota_check_at = VALUES(next_quota_check_at),
                        next_quota_check_reason = VALUES(next_quota_check_reason),
                        added_at = VALUES(added_at),
                        updated_at = VALUES(updated_at)
                    """,
                    (
                        str(payload.get("id") or ""),
                        str(payload.get("name") or ""),
                        str(payload.get("accessToken") or ""),
                        str(payload.get("refreshToken") or ""),
                        str(payload.get("utdid") or ""),
                        int(payload.get("fillPriority") or 0),
                        payload.get("expiresAt"),
                        payload.get("cookie"),
                        bool(payload.get("manualEnabled", True)),
                        bool(payload.get("autoDisabled", False)),
                        payload.get("autoDisabledReason"),
                        payload.get("lastQuotaCheckAt"),
                        payload.get("nextQuotaCheckAt"),
                        payload.get("nextQuotaCheckReason"),
                        str(payload.get("addedAt") or ""),
                        str(payload.get("updatedAt") or ""),
                    ),
                )

    def delete_account(self, account_id: str) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                deleted_count = cursor.execute(
                    "DELETE FROM accio_accounts WHERE id = %s",
                    (account_id,),
                )
        return deleted_count > 0


class MySQLPanelSettingsStore:
    def __init__(self, gateway: MySQLGateway):
        self.gateway = gateway
        self._lock = threading.RLock()
        self._cache: PanelSettings | None = None
        self.gateway.ensure_schema()

    def bootstrap_from_file_if_empty(self, file_store) -> None:
        with self._lock:
            if self.gateway.fetch_panel_settings() is not None:
                return
            self.save(file_store.load())

    def _warm_cache(self) -> PanelSettings:
        payload = self.gateway.fetch_panel_settings()
        settings, changed = load_panel_settings(payload)
        if changed or payload is None:
            settings = normalize_panel_settings(settings)
            self.gateway.save_panel_settings(settings.to_dict())
        self._cache = settings
        return settings

    def load(self) -> PanelSettings:
        with self._lock:
            if self._cache is not None:
                return self._cache
            return self._warm_cache()

    def save(self, settings: PanelSettings) -> PanelSettings:
        with self._lock:
            settings = normalize_panel_settings(settings)
            self._cache = settings
            self.gateway.save_panel_settings(settings.to_dict())
            return settings


class MySQLAccountStore(BaseAccountStore):
    def __init__(self, gateway: MySQLGateway):
        self.gateway = gateway
        self._accounts_cache: list[Account] | None = None
        self._accounts_by_id: dict[str, Account] = {}
        super().__init__()

    def _ensure_storage(self) -> None:
        self.gateway.ensure_schema()

    def _rebuild_index(self, accounts: list[Account]) -> None:
        self._accounts_by_id = {a.id: a for a in accounts}

    def _warm_cache(self) -> list[Account]:
        accounts = [
            Account.from_dict(payload) for payload in self.gateway.list_accounts()
        ]
        self._accounts_cache = accounts
        self._rebuild_index(accounts)
        return accounts

    def _read_all_unlocked(self) -> list[Account]:
        if self._accounts_cache is not None:
            return list(self._accounts_cache)
        return self._warm_cache()

    def _get_account_unlocked(self, account_id: str) -> Account | None:
        if self._accounts_cache is None:
            self._warm_cache()
        return self._accounts_by_id.get(account_id)

    def _write_account_unlocked(self, account: Account) -> None:
        self._normalize_account(account)
        # Update cache
        if self._accounts_cache is not None:
            existing = self._accounts_by_id.get(account.id)
            if existing is not None:
                idx = self._accounts_cache.index(existing)
                self._accounts_cache[idx] = account
            else:
                self._accounts_cache.append(account)
            self._accounts_by_id[account.id] = account
        # Write-through to MySQL
        self.gateway.upsert_account(account.to_dict())

    def _delete_account_unlocked(self, account_id: str) -> bool:
        # Update cache
        if self._accounts_cache is not None:
            existing = self._accounts_by_id.pop(account_id, None)
            if existing is not None:
                self._accounts_cache.remove(existing)
        # Write-through to MySQL
        return self.gateway.delete_account(account_id)

    def bootstrap_from_file_if_empty(self, file_store) -> None:
        with self._lock:
            if self.gateway.count_accounts() > 0:
                return
            for account in file_store.list_accounts():
                self._write_account_unlocked(account)
            # Invalidate cache so next read picks up bootstrapped data
            self._accounts_cache = None
            self._accounts_by_id = {}


def _parse_database_url(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme not in _MYSQL_SCHEMES:
        raise ValueError("ACCIO_MYSQL 仅支持 mysql:// 或 mysql+pymysql://")

    host = parsed.hostname or ""
    database = parsed.path.lstrip("/")
    user = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    port = parsed.port or 3306
    charset = parse_qs(parsed.query).get("charset", ["utf8mb4"])[0] or "utf8mb4"

    if not host or not database or not user:
        raise ValueError("ACCIO_MYSQL 缺少 host、database 或 user")

    qs = parse_qs(parsed.query)
    _yes = ("true", "1", "yes")
    use_ssl = (
        qs.get("ssl", [""])[0].lower() in _yes
        or qs.get("tls", [""])[0].lower() in _yes
    )
    ssl_ca = qs.get("ssl_ca", [""])[0]

    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
        "charset": charset,
        "use_ssl": use_ssl or bool(ssl_ca),
        "ssl_ca": ssl_ca,
    }


def _account_row_to_payload(row: dict[str, Any]) -> dict[str, object]:
    return {
        "id": str(row.get("id") or ""),
        "name": str(row.get("name") or ""),
        "accessToken": str(row.get("access_token") or ""),
        "refreshToken": str(row.get("refresh_token") or ""),
        "utdid": str(row.get("utdid") or ""),
        "fillPriority": int(row.get("fill_priority") or 0),
        "expiresAt": row.get("expires_at"),
        "cookie": row.get("cookie"),
        "manualEnabled": bool(row.get("manual_enabled", True)),
        "autoDisabled": bool(row.get("auto_disabled", False)),
        "autoDisabledReason": row.get("auto_disabled_reason"),
        "lastQuotaCheckAt": row.get("last_quota_check_at"),
        "nextQuotaCheckAt": row.get("next_quota_check_at"),
        "nextQuotaCheckReason": row.get("next_quota_check_reason"),
        "addedAt": str(row.get("added_at") or ""),
        "updatedAt": str(row.get("updated_at") or ""),
    }
