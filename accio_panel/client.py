from __future__ import annotations

import secrets
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import unquote, urlencode

import requests

from .config import Settings
from .models import Account


class AccioClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._session = requests.Session()

    def get_proxies(self, proxy_url: str | None = None) -> dict[str, str] | None:
        if not proxy_url:
            return None
        return {
            "http": proxy_url,
            "https": proxy_url,
        }

    def get_headers(
        self,
        utdid: str,
        *,
        accept: str | None = None,
        cna: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "x-language": "zh",
            "x-utdid": utdid,
            "x-app-version": self.settings.version,
            "x-os": "win32",
            "accept": accept or "application/json, text/plain, */*",
        }
        if cna:
            headers["x-cna"] = cna
        if user_agent:
            headers["user-agent"] = user_agent
        return headers

    def _extract_cookie_value(self, cookie_text: str | None, key: str) -> str | None:
        if not cookie_text:
            return None
        normalized = str(cookie_text)
        for _ in range(2):
            decoded = unquote(normalized)
            if decoded == normalized:
                break
            normalized = decoded
        for part in normalized.split(";"):
            name, separator, value = part.strip().partition("=")
            if separator and name.strip() == key:
                return value.strip() or None
        return None

    def _build_activation_body(self, account: Account) -> dict[str, Any]:
        return {
            "utdid": account.utdid,
            "version": self.settings.version,
            "accessToken": account.access_token,
        }

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        proxy_url: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = self._session.request(
                method,
                url,
                timeout=self.settings.request_timeout,
                proxies=self.get_proxies(proxy_url),
                **kwargs,
            )
        except requests.RequestException as exc:
            return {"success": False, "message": str(exc)}

        try:
            payload = response.json()
        except ValueError:
            payload = {
                "success": False,
                "message": f"HTTP {response.status_code}: {response.text[:200]}",
            }

        if isinstance(payload, dict):
            if not response.ok:
                payload["success"] = False
                payload.setdefault("message", f"HTTP {response.status_code}")
            return payload

        return {
            "success": response.ok,
            "data": payload,
            "message": "" if response.ok else f"HTTP {response.status_code}",
        }

    def build_login_url(self, callback_url: str, *, state: str | None = None) -> str:
        query = urlencode(
            {
                "return_url": callback_url,
                "state": state or secrets.token_hex(32),
            }
        )
        return f"https://www.accio.com/login?{query}"

    def query_quota(
        self,
        account: Account,
        *,
        proxy_url: str | None = None,
    ) -> dict[str, Any]:
        params = {
            "accessToken": account.access_token,
            "utdid": account.utdid,
            "version": self.settings.version,
        }
        return self._request_json(
            "GET",
            f"{self.settings.base_url}/api/entitlement/currentSubscription",
            params=params,
            headers={
                **self.get_headers(
                    account.utdid,
                    accept="*/*",
                    cna=self._extract_cookie_value(account.cookie, "cna"),
                    user_agent="node",
                ),
                "accept-language": "*",
                "sec-fetch-mode": "cors",
            },
            proxy_url=proxy_url,
        )

    def refresh_token(
        self,
        account: Account,
        *,
        proxy_url: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "utdid": account.utdid,
            "version": self.settings.version,
            "accessToken": account.access_token,
            "refreshToken": account.refresh_token,
        }
        return self._request_json(
            "POST",
            f"{self.settings.base_url}/api/auth/refresh_token",
            json=body,
            headers=self.get_headers(account.utdid),
            proxy_url=proxy_url,
        )

    def query_userinfo(
        self,
        account: Account,
        *,
        proxy_url: str | None = None,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"{self.settings.base_url}/api/auth/userinfo",
            json=self._build_activation_body(account),
            headers=self.get_headers(
                account.utdid,
                accept="*/*",
                cna=self._extract_cookie_value(account.cookie, "cna"),
                user_agent="node",
            ),
            proxy_url=proxy_url,
        )

    def query_invitation(
        self,
        account: Account,
        *,
        proxy_url: str | None = None,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"{self.settings.base_url}/api/invitation/query",
            json=self._build_activation_body(account),
            headers=self.get_headers(
                account.utdid,
                accept="*/*",
                cna=self._extract_cookie_value(account.cookie, "cna"),
                user_agent="node",
            ),
            proxy_url=proxy_url,
        )

    def query_channel(
        self,
        account: Account,
        *,
        proxy_url: str | None = None,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"{self.settings.base_url}/api/channel/query",
            json={"accessToken": account.access_token},
            headers={
                "content-type": "application/json",
                "accept": "*/*",
                "user-agent": "node",
            },
            proxy_url=proxy_url,
        )

    def query_llm_config(
        self,
        account: Account,
        *,
        proxy_url: str | None = None,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"{self.settings.base_url}/api/llm/config",
            json={"token": account.access_token},
            headers={
                "content-type": "application/json",
                "accept": "application/json",
                "user-agent": "node",
            },
            proxy_url=proxy_url,
        )

    def activate_account(
        self,
        account: Account,
        *,
        proxy_url: str | None = None,
    ) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_userinfo = executor.submit(self.query_userinfo, account, proxy_url=proxy_url)
            f_invitation = executor.submit(self.query_invitation, account, proxy_url=proxy_url)
            f_channel = executor.submit(self.query_channel, account, proxy_url=proxy_url)
            userinfo = f_userinfo.result()
            invitation = f_invitation.result()
            channel = f_channel.result()

        userinfo_success = bool(userinfo.get("success"))
        invitation_success = bool(invitation.get("success"))
        channel_success = bool(channel.get("success"))
        required_success = userinfo_success and invitation_success

        if required_success and channel_success:
            message = "账号激活完成"
        elif required_success:
            message = "账号激活完成，渠道查询未成功"
        else:
            message = "账号激活未完成，请检查激活步骤结果"

        userinfo_data = userinfo.get("data") if isinstance(userinfo.get("data"), dict) else {}
        channel_data = channel.get("data") if isinstance(channel.get("data"), dict) else {}

        return {
            "success": required_success,
            "message": message,
            "userName": str(userinfo_data.get("userName") or ""),
            "userId": str(userinfo_data.get("userId") or ""),
            "accioId": str(userinfo_data.get("accioId") or ""),
            "invitationGranted": invitation.get("data"),
            "channelAuthorizations": channel_data.get("authorizations")
            if isinstance(channel_data.get("authorizations"), list)
            else [],
            "steps": [
                {
                    "key": "userinfo",
                    "label": "用户信息",
                    "success": userinfo_success,
                    "optional": False,
                    "message": str(userinfo.get("message") or ""),
                    "code": str(userinfo.get("code") or ""),
                },
                {
                    "key": "invitation",
                    "label": "邀请状态",
                    "success": invitation_success,
                    "optional": False,
                    "message": str(invitation.get("message") or ""),
                    "code": str(invitation.get("code") or ""),
                    "data": invitation.get("data"),
                },
                {
                    "key": "channel",
                    "label": "渠道信息",
                    "success": channel_success,
                    "optional": True,
                    "message": str(channel.get("message") or ""),
                    "code": str(channel.get("code") or ""),
                    "data": channel.get("data"),
                },
            ],
        }

    def generate_content(
        self,
        account: Account,
        body: dict[str, Any],
        *,
        proxy_url: str | None = None,
    ) -> requests.Response:
        return self._session.post(
            f"{self.settings.base_url}/api/adk/llm/generateContent",
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "utdid": account.utdid,
                "version": self.settings.version,
                "appKey": self.settings.app_key,
                "user-agent": "node",
            },
            proxies=self.get_proxies(proxy_url),
            stream=True,
            timeout=(self.settings.request_timeout, 300),
        )
