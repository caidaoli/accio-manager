from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from .models import Account, normalize_timestamp, now_text
from .utils import new_utdid


class AccountStore:
    def __init__(self, accounts_dir: Path, legacy_file_path: Path | None = None):
        self.accounts_dir = accounts_dir
        self.legacy_file_path = legacy_file_path
        self._lock = threading.RLock()
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self.accounts_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_if_needed()

    def _account_file(self, account_id: str) -> Path:
        return self.accounts_dir / f"{account_id}.json"

    def _list_account_files_unlocked(self) -> list[Path]:
        self.accounts_dir.mkdir(parents=True, exist_ok=True)
        return sorted(
            (
                path
                for path in self.accounts_dir.glob("*.json")
                if path.is_file()
            ),
            key=lambda item: item.name,
        )

    def _normalize_account(self, account: Account) -> bool:
        changed = False
        if not account.id:
            account.id = uuid.uuid4().hex
            changed = True
        if not account.utdid:
            account.utdid = new_utdid()
            changed = True
        if account.expires_at is not None:
            normalized_expires_at = normalize_timestamp(account.expires_at)
            if normalized_expires_at != account.expires_at:
                account.expires_at = normalized_expires_at
                changed = True
        return changed

    def _load_account_file_unlocked(self, file_path: Path) -> Account | None:
        try:
            raw = file_path.read_text(encoding="utf-8").strip() or "{}"
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        account = Account.from_dict(payload)
        if self._normalize_account(account):
            self._write_account_unlocked(account)
        return account

    def _write_account_unlocked(self, account: Account) -> None:
        self._normalize_account(account)
        self._account_file(account.id).write_text(
            json.dumps(account.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _delete_account_unlocked(self, account_id: str) -> bool:
        account_file = self._account_file(account_id)
        if not account_file.exists():
            return False
        account_file.unlink()
        return True

    def _load_legacy_accounts_unlocked(self) -> list[Account]:
        if not self.legacy_file_path or not self.legacy_file_path.exists():
            return []

        raw = self.legacy_file_path.read_text(encoding="utf-8").strip() or "[]"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = []

        if not isinstance(payload, list):
            payload = []

        accounts: list[Account] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            account = Account.from_dict(item)
            self._normalize_account(account)
            accounts.append(account)
        return accounts

    def _migrate_legacy_if_needed(self) -> None:
        if not self.legacy_file_path or not self.legacy_file_path.exists():
            return

        if self._list_account_files_unlocked():
            return

        for account in self._load_legacy_accounts_unlocked():
            self._write_account_unlocked(account)

    def _read_all_unlocked(self) -> list[Account]:
        self._ensure_storage()
        accounts = [
            account
            for account in (
                self._load_account_file_unlocked(file_path)
                for file_path in self._list_account_files_unlocked()
            )
            if account is not None
        ]
        accounts.sort(key=lambda item: (item.added_at, item.name, item.id))
        return accounts

    def list_accounts(self) -> list[Account]:
        with self._lock:
            return self._read_all_unlocked()

    def get_account(self, account_id: str) -> Account | None:
        with self._lock:
            return self._load_account_file_unlocked(self._account_file(account_id))

    def save(self, account: Account) -> Account:
        with self._lock:
            self._write_account_unlocked(account)
            return account

    def _next_account_name(self, accounts: list[Account]) -> str:
        max_index = 0
        for account in accounts:
            if account.name.startswith("账号"):
                suffix = account.name[2:]
                if suffix.isdigit():
                    max_index = max(max_index, int(suffix))
        return f"账号{max_index + 1}"

    def upsert_from_callback(
        self,
        *,
        access_token: str,
        refresh_token: str,
        expires_at: str | int | None,
        cookie: str | None,
    ) -> tuple[Account, bool]:
        with self._lock:
            accounts = self._read_all_unlocked()
            now = now_text()

            for account in accounts:
                if account.access_token == access_token:
                    account.refresh_token = refresh_token
                    account.expires_at = normalize_timestamp(expires_at)
                    account.cookie = cookie or account.cookie
                    account.updated_at = now
                    self._write_account_unlocked(account)
                    return account, False

                if account.refresh_token == refresh_token:
                    account.access_token = access_token
                    account.expires_at = normalize_timestamp(expires_at)
                    account.cookie = cookie or account.cookie
                    account.updated_at = now
                    self._write_account_unlocked(account)
                    return account, False

                if cookie and account.cookie and account.cookie == cookie:
                    account.access_token = access_token
                    account.refresh_token = refresh_token
                    account.expires_at = normalize_timestamp(expires_at)
                    account.updated_at = now
                    self._write_account_unlocked(account)
                    return account, False

            account = Account(
                id=uuid.uuid4().hex,
                name=self._next_account_name(accounts),
                access_token=access_token,
                refresh_token=refresh_token,
                utdid=new_utdid(),
                expires_at=normalize_timestamp(expires_at),
                cookie=cookie,
                added_at=now,
                updated_at=now,
            )
            self._write_account_unlocked(account)
            return account, True

    def update_tokens(
        self,
        account_id: str,
        *,
        access_token: str,
        refresh_token: str,
        expires_at: str | int | None,
    ) -> Account | None:
        with self._lock:
            account = self.get_account(account_id)
            if not account:
                return None
            account.access_token = access_token
            account.refresh_token = refresh_token
            account.expires_at = normalize_timestamp(expires_at)
            account.updated_at = now_text()
            self._write_account_unlocked(account)
            return account

    def rename(self, account_id: str, name: str) -> Account | None:
        with self._lock:
            account = self.get_account(account_id)
            if not account:
                return None
            account.name = name
            account.updated_at = now_text()
            self._write_account_unlocked(account)
            return account

    def set_manual_enabled(self, account_id: str, enabled: bool) -> Account | None:
        with self._lock:
            account = self.get_account(account_id)
            if not account:
                return None
            account.manual_enabled = enabled
            account.updated_at = now_text()
            self._write_account_unlocked(account)
            return account

    def set_auto_disabled(
        self,
        account_id: str,
        auto_disabled: bool,
        reason: str | None = None,
    ) -> Account | None:
        with self._lock:
            account = self.get_account(account_id)
            if not account:
                return None
            account.auto_disabled = auto_disabled
            account.auto_disabled_reason = reason if auto_disabled else None
            account.updated_at = now_text()
            self._write_account_unlocked(account)
            return account

    def delete(self, account_id: str) -> bool:
        with self._lock:
            return self._delete_account_unlocked(account_id)
