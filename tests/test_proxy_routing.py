import json
import tempfile
import unittest
import asyncio
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request

from accio_panel.app_settings import PanelSettings
from accio_panel.config import Settings
from accio_panel.models import Account
from accio_panel.mysql_storage import MySQLGateway
from accio_panel.store import AccountStore
from accio_panel.web import _ordered_proxy_candidates, create_app


class _FakeSSEUpstreamResponse:
    def __init__(self, payloads: list[dict[str, object]], status_code: int = 200):
        self._payloads = list(payloads)
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = json.dumps(payloads, ensure_ascii=False)
        self.closed = False

    def iter_lines(self, decode_unicode: bool = False):
        for payload in self._payloads:
            line = f"data: {json.dumps(payload, ensure_ascii=False)}"
            if decode_unicode:
                yield line
            else:
                yield line.encode("utf-8")

    def close(self):
        self.closed = True


class _FakeProxyClient:
    def __init__(self, responses_by_account_id: dict[str, list[_FakeSSEUpstreamResponse]]):
        self._responses_by_account_id = {
            account_id: list(responses)
            for account_id, responses in responses_by_account_id.items()
        }
        self.calls: list[str] = []

    def query_llm_config(self, account, proxy_url=""):
        return {"success": True, "data": {"models": []}, "message": ""}

    def refresh_token(self, account, proxy_url=""):
        return {"success": False, "message": "not implemented in test"}

    def generate_content(self, account, body, proxy_url=""):
        self.calls.append(account.id)
        responses = self._responses_by_account_id.get(account.id, [])
        if not responses:
            raise AssertionError(f"账号 {account.id} 没有预设上游响应")
        return responses.pop(0)


class _RecordingCursor:
    def __init__(self, rows: list[dict[str, object]] | None = None):
        self.rows = list(rows or [])
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def execute(self, sql: str, params: tuple[object, ...] | None = None):
        self.executed.append((sql, params))
        return 1

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _RecordingConnection:
    def __init__(self, cursor: _RecordingCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class _RecordingGateway(MySQLGateway):
    def __init__(self, cursor: _RecordingCursor):
        super().__init__(
            host="127.0.0.1",
            port=3306,
            user="root",
            password="secret",
            database="accio",
        )
        self._recording_connection = _RecordingConnection(cursor)

    @contextmanager
    def _connect(self):
        yield self._recording_connection


def _wrapped_event(event: dict[str, object]) -> dict[str, object]:
    return {"raw_response_json": event}


def _empty_claude_stream() -> _FakeSSEUpstreamResponse:
    return _FakeSSEUpstreamResponse(
        [
            _wrapped_event(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg-empty",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "usage": {"input_tokens": 5, "output_tokens": 0},
                    },
                }
            ),
            _wrapped_event(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                }
            ),
            _wrapped_event({"type": "message_stop"}),
        ]
    )


def _text_claude_stream(text: str) -> _FakeSSEUpstreamResponse:
    return _FakeSSEUpstreamResponse(
        [
            _wrapped_event(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg-text",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "usage": {"input_tokens": 7, "output_tokens": len(text)},
                    },
                }
            ),
            _wrapped_event(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            ),
            _wrapped_event(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                }
            ),
            _wrapped_event({"type": "content_block_stop", "index": 0}),
            _wrapped_event(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": len(text)},
                }
            ),
            _wrapped_event({"type": "message_stop"}),
        ]
    )


async def _invoke_openai_chat_route(
    app,
    *,
    headers: dict[str, str],
    payload: dict[str, object],
):
    route = next(
        route
        for route in app.router.routes
        if getattr(route, "path", "") == "/v1/chat/completions"
        and "POST" in getattr(route, "methods", set())
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    received = False

    async def receive():
        nonlocal received
        if received:
            return {"type": "http.request", "body": b"", "more_body": False}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "headers": [
            (key.lower().encode("utf-8"), value.encode("utf-8"))
            for key, value in headers.items()
        ],
        "query_string": b"",
        "scheme": "http",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
        "app": app,
    }
    request = Request(scope, receive)
    response = await route.endpoint(request)
    body_chunks: list[bytes] = []
    if hasattr(response, "body_iterator"):
        async for chunk in response.body_iterator:
            body_chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
    else:
        body_chunks.append(response.body)
    return response, b"".join(body_chunks).decode("utf-8")


class ProxyRoutingTests(unittest.TestCase):
    def test_ordered_proxy_candidates_excludes_auto_disabled_and_model_disabled_accounts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AccountStore(Path(temp_dir) / "accounts")
            store.save(
                Account(
                    id="acc-enabled",
                    name="可用账号",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                )
            )
            store.save(
                Account(
                    id="acc-auto-disabled",
                    name="自动禁用账号",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    auto_disabled=True,
                    auto_disabled_reason="空回复",
                )
            )
            store.save(
                Account(
                    id="acc-model-disabled",
                    name="模型禁用账号",
                    access_token="token-3",
                    refresh_token="refresh-3",
                    utdid="utdid-3",
                    disabled_models={
                        "claude-opus-4-6": "模型 claude-opus-4-6 出现空回复，已自动禁用该账号调用此模型。"
                    },
                )
            )

            candidates = _ordered_proxy_candidates(store, "claude-opus-4-6")

            self.assertEqual([account.id for account in candidates], ["acc-enabled"])

    def test_mysql_gateway_roundtrip_includes_disabled_models_column(self):
        disabled_reason = "模型 claude-opus-4-6 出现空回复"
        row_cursor = _RecordingCursor(
            rows=[
                {
                    "id": "acc-1",
                    "name": "账号1",
                    "access_token": "token-1",
                    "refresh_token": "refresh-1",
                    "utdid": "utdid-1",
                    "fill_priority": 10,
                    "expires_at": None,
                    "cookie": None,
                    "manual_enabled": 1,
                    "auto_disabled": 0,
                    "auto_disabled_reason": None,
                    "last_quota_check_at": None,
                    "next_quota_check_at": None,
                    "next_quota_check_reason": None,
                    "disabled_models": json.dumps(
                        {"claude-opus-4-6": disabled_reason},
                        ensure_ascii=False,
                    ),
                    "added_at": "2026-04-05 00:00:00",
                    "updated_at": "2026-04-05 00:00:00",
                }
            ]
        )
        gateway = _RecordingGateway(row_cursor)

        accounts = gateway.list_accounts()

        self.assertEqual(
            accounts[0]["disabledModels"],
            {"claude-opus-4-6": disabled_reason},
        )
        select_sql = row_cursor.executed[0][0].lower()
        self.assertIn("disabled_models", select_sql)

        write_cursor = _RecordingCursor()
        write_gateway = _RecordingGateway(write_cursor)
        write_gateway.upsert_account(
            Account(
                id="acc-1",
                name="账号1",
                access_token="token-1",
                refresh_token="refresh-1",
                utdid="utdid-1",
                disabled_models={"claude-opus-4-6": disabled_reason},
            ).to_dict()
        )

        insert_sql, params = write_cursor.executed[0]
        self.assertIn("disabled_models", insert_sql.lower())
        self.assertIsNotNone(params)
        self.assertIn(
            json.dumps({"claude-opus-4-6": disabled_reason}, ensure_ascii=False),
            params,
        )

    def test_openai_stream_retries_once_after_empty_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_empty_claude_stream()],
                    "acc-2": [_text_claude_stream("第二个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.web._is_allowed_dynamic_model", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(admin_password="admin", session_secret="test-session")
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                )
            )

            response, response_text = asyncio.run(
                _invoke_openai_chat_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-opus-4-6",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-2")
            self.assertIn("第二个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2"])
            disabled_account = store.get_account("acc-1")
            self.assertIsNotNone(disabled_account)
            self.assertIn("claude-opus-4-6", disabled_account.disabled_models)


if __name__ == "__main__":
    unittest.main()
