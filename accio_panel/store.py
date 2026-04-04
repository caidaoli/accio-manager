from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

from .models import Account, normalize_fill_priority, normalize_timestamp, now_text
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
        normalized_fill_priority = normalize_fill_priority(account.fill_priority)
        if normalized_fill_priority != account.fill_priority:
            account.fill_priority = normalized_fill_priority
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

    def _match_existing_account_unlocked(
        self,
        accounts: list[Account],
        imported: Account,
    ) -> Account | None:
        for account in accounts:
            if imported.id and account.id == imported.id:
                return account
            if imported.access_token and account.access_token == imported.access_token:
                return account
            if imported.refresh_token and account.refresh_token == imported.refresh_token:
                return account
            if imported.cookie and account.cookie and account.cookie == imported.cookie:
                return account
        return None

    def import_accounts(
        self,
        payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with self._lock:
            accounts = self._read_all_unlocked()
            created_count = 0
            updated_count = 0
            failures: list[str] = []

            for index, payload in enumerate(payloads, start=1):
                try:
                    imported = Account.from_dict(payload)
                    self._normalize_account(imported)
                except Exception:
                    failures.append(f"第 {index} 项账号数据格式无效")
                    continue

                if not imported.access_token:
                    failures.append(f"{imported.name}: 缺少 accessToken")
                    continue
                if not imported.refresh_token:
                    failures.append(f"{imported.name}: 缺少 refreshToken")
                    continue

                now = now_text()
                imported.expires_at = normalize_timestamp(imported.expires_at)
                if not imported.utdid:
                    imported.utdid = new_utdid()
                if not imported.name or imported.name == "未命名账号":
                    imported.name = self._next_account_name(accounts)
                if not imported.added_at:
                    imported.added_at = now

                existing = self._match_existing_account_unlocked(accounts, imported)
                if existing:
                    existing.name = imported.name or existing.name
                    existing.access_token = imported.access_token
                    existing.refresh_token = imported.refresh_token
                    existing.utdid = imported.utdid or existing.utdid or new_utdid()
                    existing.expires_at = imported.expires_at
                    existing.cookie = imported.cookie or existing.cookie
                    existing.manual_enabled = imported.manual_enabled
                    existing.auto_disabled = imported.auto_disabled
                    existing.auto_disabled_reason = (
                        imported.auto_disabled_reason if imported.auto_disabled else None
                    )
                    existing.last_quota_check_at = imported.last_quota_check_at
                    existing.next_quota_check_at = imported.next_quota_check_at
                    existing.next_quota_check_reason = imported.next_quota_check_reason
                    existing.added_at = imported.added_at or existing.added_at
                    existing.updated_at = now
                    self._write_account_unlocked(existing)
                    updated_count += 1
                    continue

                while self._account_file(imported.id).exists():
                    imported.id = uuid.uuid4().hex

                imported.updated_at = now
                self._write_account_unlocked(imported)
                accounts.append(imported)
                created_count += 1

            return {
                "createdCount": created_count,
                "updatedCount": updated_count,
                "failureCount": len(failures),
                "importedCount": created_count + updated_count,
                "failures": failures,
            }

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

    def set_fill_priority(self, account_id: str, fill_priority: int) -> Account | None:
        with self._lock:
            account = self.get_account(account_id)
            if not account:
                return None
            account.fill_priority = normalize_fill_priority(fill_priority)
            account.updated_at = now_text()
            self._write_account_unlocked(account)
            return account

    def set_manual_enabled(self, account_id: str, enabled: bool) -> Account | None:
        with self._lock:
            account = self.get_account(account_id)
            if not account:
                return None
            account.manual_enabled = enabled
            if enabled and not account.auto_disabled:
                account.auto_disabled_reason = None
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

    def _is_abnormal_auto_disabled_unlocked(self, account: Account) -> bool:
        reason_text = str(account.auto_disabled_reason or "").strip()
        if not reason_text:
            return False

        normalized_reason = reason_text.lower()
        is_abnormal_reason = (
            "auth not pass" in normalized_reason
            or "请手动处理" in reason_text
        )
        if not is_abnormal_reason:
            return False

        # 账号当前处于自动禁用状态
        if account.auto_disabled:
            return True
        # 账号已被手动禁用（manualEnabled=false），但 reason 仍残留异常信息
        if not account.manual_enabled:
            return True
        return False

    def list_abnormal_auto_disabled_accounts(self) -> list[Account]:
        with self._lock:
            return [
                account
                for account in self._read_all_unlocked()
                if self._is_abnormal_auto_disabled_unlocked(account)
            ]

    def delete_abnormal_auto_disabled_accounts(self) -> dict[str, Any]:
        with self._lock:
            accounts = self._read_all_unlocked()
            matched_accounts = [
                account
                for account in accounts
                if self._is_abnormal_auto_disabled_unlocked(account)
            ]
            deleted_ids: list[str] = []
            deleted_names: list[str] = []
            failures: list[str] = []

            for account in matched_accounts:
                if self._delete_account_unlocked(account.id):
                    deleted_ids.append(account.id)
                    deleted_names.append(account.name)
                    continue
                failures.append(f"{account.name}: 删除失败")

            return {
                "processedCount": len(matched_accounts),
                "deletedCount": len(deleted_ids),
                "failureCount": len(failures),
                "deletedIds": deleted_ids,
                "deletedNames": deleted_names,
                "failures": failures,
            }

    def delete(self, account_id: str) -> bool:
        with self._lock:
            return self._delete_account_unlocked(account_id)
