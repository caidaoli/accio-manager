from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import requests

from .config import Settings
from .models import Account


class AccioClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def get_proxies(self, proxy_url: str | None = None) -> dict[str, str] | None:
        if not proxy_url:
            return None
        return {
            "http": proxy_url,
            "https": proxy_url,
        }

    def get_headers(self, utdid: str) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "x-language": "zh",
            "x-utdid": utdid,
            "x-app-version": self.settings.version,
            "x-os": "win32",
            "accept": "application/json, text/plain, */*",
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
            response = requests.request(
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

    def build_login_url(self, callback_url: str) -> str:
        query = urlencode(
            {
                "return_url": callback_url,
                "state": "accio-panel",
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
            f"{self.settings.base_url}/api/entitlement/quota",
            params=params,
            headers=self.get_headers(account.utdid),
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

    def generate_content(
        self,
        account: Account,
        body: dict[str, Any],
        *,
        proxy_url: str | None = None,
    ) -> requests.Response:
        return requests.post(
            f"{self.settings.base_url}/api/adk/llm/generateContent",
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "utdid": account.utdid,
                "version": self.settings.version,
                "user-agent": "node",
            },
            proxies=self.get_proxies(proxy_url),
            stream=True,
            timeout=(self.settings.request_timeout, 300),
        )
