import json
import tempfile
import unittest
import asyncio
import warnings
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request

from accio_panel.app_settings import PanelSettings
from accio_panel.config import Settings
from accio_panel.models import Account
from accio_panel.mysql_storage import MySQLGateway
from accio_panel.store import AccountStore
from accio_panel.proxy_selection import _ordered_proxy_candidates, _select_proxy_account
from accio_panel.web import create_app


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


class _FakeHTTPUpstreamResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = text
        self.closed = False

    def close(self):
        self.closed = True


class _FakeProxyClient:
    def __init__(self, responses_by_account_id: dict[str, list[_FakeSSEUpstreamResponse]]):
        self._responses_by_account_id = {
            account_id: list(responses)
            for account_id, responses in responses_by_account_id.items()
        }
        self.calls: list[str] = []
        self.request_bodies: list[dict[str, object]] = []

    def query_llm_config(self, account, proxy_url=""):
        return {"success": True, "data": {"models": []}, "message": ""}

    def refresh_token(self, account, proxy_url=""):
        return {"success": False, "message": "not implemented in test"}

    def generate_content(self, account, body, proxy_url=""):
        self.calls.append(account.id)
        self.request_bodies.append(dict(body))
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


def _gemini_text_response(text: str) -> _FakeSSEUpstreamResponse:
    return _FakeSSEUpstreamResponse(
        [
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": text}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": len(text),
                    "totalTokenCount": len(text) + 1,
                },
            }
        ]
    )


def _upstream_turn_error_stream(
    code: str = "505",
    message: str = "internal server error",
) -> _FakeSSEUpstreamResponse:
    return _FakeSSEUpstreamResponse(
        [
            {
                "turn_complete": True,
                "error_code": code,
                "error_message": message,
            }
        ]
    )


def _quota_exhausted_http_error_response() -> _FakeHTTPUpstreamResponse:
    return _FakeHTTPUpstreamResponse(
        429,
        json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "quota exhausted",
                },
            },
            ensure_ascii=False,
        ),
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


async def _invoke_anthropic_messages_route(
    app,
    *,
    headers: dict[str, str],
    payload: dict[str, object],
):
    route = next(
        route
        for route in app.router.routes
        if getattr(route, "path", "") == "/v1/messages"
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
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
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


async def _invoke_native_generate_content_route(
    app,
    *,
    headers: dict[str, str],
    payload: dict[str, object],
):
    route = next(
        route
        for route in app.router.routes
        if getattr(route, "path", "") == "/api/adk/llm/generateContent"
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
        "path": "/api/adk/llm/generateContent",
        "raw_path": b"/api/adk/llm/generateContent",
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


async def _invoke_gemini_generate_content_route(
    app,
    *,
    headers: dict[str, str],
    model_name: str,
    payload: dict[str, object],
):
    route = next(
        route
        for route in app.router.routes
        if getattr(route, "path", "") == "/v1beta/models/{model_name}:generateContent"
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
        "path": f"/v1beta/models/{model_name}:generateContent",
        "raw_path": f"/v1beta/models/{model_name}:generateContent".encode("utf-8"),
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
    response = await route.endpoint(request, model_name=model_name)
    body_chunks: list[bytes] = []
    if hasattr(response, "body_iterator"):
        async for chunk in response.body_iterator:
            body_chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
    else:
        body_chunks.append(response.body)
    return response, b"".join(body_chunks).decode("utf-8")


async def _invoke_openai_responses_route(
    app,
    *,
    headers: dict[str, str],
    payload: dict[str, object],
):
    route = next(
        route
        for route in app.router.routes
        if getattr(route, "path", "") == "/v1/responses"
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
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
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


def _read_api_logs(settings: Settings) -> list[dict[str, object]]:
    if not settings.api_logs_file.exists():
        return []
    return [
        json.loads(line)
        for line in settings.api_logs_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ProxyRoutingTests(unittest.TestCase):
    def test_create_app_does_not_register_deprecated_on_event_hooks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")

            async def _noop_scheduler(_application):
                return None

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                    create_app(settings)

            self.assertEqual(
                [
                    str(item.message)
                    for item in caught
                    if "on_event is deprecated" in str(item.message)
                ],
                [],
            )

    def test_create_app_lifespan_starts_and_stops_quota_scheduler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")

            async def _exercise() -> None:
                scheduler_started = asyncio.Event()
                scheduler_cancelled = asyncio.Event()

                async def _blocking_scheduler(_application):
                    scheduler_started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        scheduler_cancelled.set()
                        raise

                with patch("accio_panel.web._quota_scheduler_loop", _blocking_scheduler):
                    app = create_app(settings)
                    self.assertIsNone(app.state.quota_scheduler_task)
                    async with app.router.lifespan_context(app):
                        task = app.state.quota_scheduler_task
                        self.assertIsNotNone(task)
                        await asyncio.wait_for(scheduler_started.wait(), timeout=1)
                        self.assertFalse(task.done())

                    await asyncio.wait_for(scheduler_cancelled.wait(), timeout=1)
                self.assertIsNone(app.state.quota_scheduler_task)

            asyncio.run(_exercise())

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
                    "last_remaining_quota": 17,
                    "last_total_quota": 20,
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
        self.assertEqual(accounts[0]["lastRemainingQuota"], 17)
        self.assertEqual(accounts[0]["lastTotalQuota"], 20)
        select_sql = row_cursor.executed[0][0].lower()
        self.assertIn("disabled_models", select_sql)
        self.assertIn("last_remaining_quota", select_sql)
        self.assertIn("last_total_quota", select_sql)

        write_cursor = _RecordingCursor()
        write_gateway = _RecordingGateway(write_cursor)
        write_gateway.upsert_account(
            Account(
                id="acc-1",
                name="账号1",
                access_token="token-1",
                refresh_token="refresh-1",
                utdid="utdid-1",
                last_remaining_quota=17,
                last_total_quota=20,
                disabled_models={"claude-opus-4-6": disabled_reason},
            ).to_dict()
        )

        insert_sql, params = write_cursor.executed[0]
        self.assertIn("disabled_models", insert_sql.lower())
        self.assertIn("last_remaining_quota", insert_sql.lower())
        self.assertIn("last_total_quota", insert_sql.lower())
        self.assertIsNotNone(params)
        self.assertIn(
            json.dumps({"claude-opus-4-6": disabled_reason}, ensure_ascii=False),
            params,
        )
        self.assertIn(17, params)
        self.assertIn(20, params)

    def test_select_proxy_account_can_exclude_current_account(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                app = create_app(settings)

            panel_settings = PanelSettings(
                admin_password="admin",
                session_secret="test-session",
                api_account_strategy="fill",
            )
            app.state.panel_settings_store.save(panel_settings)
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    fill_priority=10,
                )
            )

            selected_account, _ = _select_proxy_account(
                app,
                panel_settings,
                "claude-sonnet-4-6",
                exclude_account_ids={"acc-1"},
            )

            self.assertEqual(selected_account.id, "acc-2")

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
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
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
                        "model": "deepseek-r1",
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
            self.assertIn("deepseek-r1", disabled_account.disabled_models)
            attempt_logs = [
                item
                for item in _read_api_logs(settings)
                if item.get("phase") == "upstream_attempt"
            ]
            self.assertEqual(len(attempt_logs), 2)
            self.assertEqual(
                [
                    (
                        item.get("event"),
                        item.get("attempt"),
                        item.get("accountId"),
                        item.get("success"),
                        item.get("stopReason"),
                    )
                    for item in attempt_logs
                ],
                [
                    ("v1_chat_completions", 1, "acc-1", False, "empty_response"),
                    ("v1_chat_completions", 2, "acc-2", True, "end_turn"),
                ],
            )

    def test_openai_stream_claude_retries_with_different_account_after_empty_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [
                        _empty_claude_stream(),
                        _text_claude_stream("同账号重试"),
                    ],
                    "acc-2": [_text_claude_stream("第二个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    fill_priority=10,
                )
            )

            response, response_text = asyncio.run(
                _invoke_openai_chat_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
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
            self.assertNotIn("claude-sonnet-4-6", disabled_account.disabled_models)

    def test_openai_responses_stream_retries_once_after_empty_response_and_logs_attempts(self):
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
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    fill_priority=10,
                )
            )

            response, response_text = asyncio.run(
                _invoke_openai_responses_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "stream": True,
                        "input": "hello",
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-2")
            self.assertIn("第二个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2"])
            attempt_logs = [
                item
                for item in _read_api_logs(settings)
                if item.get("phase") == "upstream_attempt"
            ]
            self.assertEqual(len(attempt_logs), 2)
            self.assertEqual(
                [
                    (
                        item.get("event"),
                        item.get("attempt"),
                        item.get("accountId"),
                        item.get("success"),
                        item.get("stopReason"),
                    )
                    for item in attempt_logs
                ],
                [
                    ("v1_responses", 1, "acc-1", False, "empty_response"),
                    ("v1_responses", 2, "acc-2", True, "end_turn"),
                ],
            )

    def test_anthropic_stream_retries_once_after_empty_response_and_logs_attempts(self):
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
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    fill_priority=10,
                )
            )

            response, response_text = asyncio.run(
                _invoke_anthropic_messages_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 256,
                        "stream": True,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-2")
            self.assertIn("第二个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2"])
            attempt_logs = [
                item
                for item in _read_api_logs(settings)
                if item.get("phase") == "upstream_attempt"
            ]
            self.assertEqual(len(attempt_logs), 2)
            self.assertEqual(
                [
                    (
                        item.get("event"),
                        item.get("attempt"),
                        item.get("accountId"),
                        item.get("success"),
                        item.get("stopReason"),
                    )
                    for item in attempt_logs
                ],
                [
                    ("v1_messages", 1, "acc-1", False, "empty_response"),
                    ("v1_messages", 2, "acc-2", True, "end_turn"),
                ],
            )

    def test_anthropic_non_stream_claude_retries_with_different_account_after_empty_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [
                        _empty_claude_stream(),
                        _text_claude_stream("同账号重试"),
                    ],
                    "acc-2": [_text_claude_stream("第二个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    fill_priority=10,
                )
            )

            response, response_text = asyncio.run(
                _invoke_anthropic_messages_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 256,
                        "stream": False,
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
            self.assertNotIn("claude-sonnet-4-6", disabled_account.disabled_models)

    def test_openai_stream_returns_error_when_upstream_turn_error_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_upstream_turn_error_stream()],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    fill_priority=10,
                )
            )

            response, response_text = asyncio.run(
                _invoke_openai_chat_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
            )

            self.assertEqual(response.status_code, 502)
            self.assertEqual(fake_client.calls, ["acc-1"])
            payload = json.loads(response_text)
            self.assertEqual(payload["error"]["code"], "upstream_error")
            self.assertIn("internal server error", payload["error"]["message"])

    def test_openai_stream_retries_once_after_retryable_upstream_turn_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [
                        _upstream_turn_error_stream(
                            code="555",
                            message="blocked by sentinel rate limit",
                        )
                    ],
                    "acc-2": [_text_claude_stream("第二个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    fill_priority=10,
                )
            )

            response, response_text = asyncio.run(
                _invoke_openai_chat_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-2")
            self.assertIn("第二个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2"])
            attempt_logs = [
                item
                for item in _read_api_logs(settings)
                if item.get("phase") == "upstream_attempt"
            ]
            self.assertEqual(len(attempt_logs), 2)
            self.assertEqual(
                [
                    (
                        item.get("event"),
                        item.get("attempt"),
                        item.get("accountId"),
                        item.get("success"),
                        item.get("stopReason"),
                    )
                    for item in attempt_logs
                ],
                [
                    ("v1_chat_completions", 1, "acc-1", False, "upstream_turn_error"),
                    ("v1_chat_completions", 2, "acc-2", True, "end_turn"),
                ],
            )

    def test_anthropic_non_stream_retries_once_after_retryable_upstream_turn_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [
                        _upstream_turn_error_stream(
                            code="555",
                            message="blocked by sentinel rate limit",
                        )
                    ],
                    "acc-2": [_text_claude_stream("第二个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    fill_priority=10,
                )
            )

            response, response_text = asyncio.run(
                _invoke_anthropic_messages_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 256,
                        "stream": False,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-2")
            self.assertIn("第二个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2"])
            attempt_logs = [
                item
                for item in _read_api_logs(settings)
                if item.get("phase") == "upstream_attempt"
            ]
            self.assertEqual(len(attempt_logs), 2)
            self.assertEqual(
                [
                    (
                        item.get("event"),
                        item.get("attempt"),
                        item.get("accountId"),
                        item.get("success"),
                        item.get("stopReason"),
                    )
                    for item in attempt_logs
                ],
                [
                    ("v1_messages", 1, "acc-1", False, "upstream_turn_error"),
                    ("v1_messages", 2, "acc-2", True, "end_turn"),
                ],
            )

    def test_anthropic_non_stream_retries_once_after_retryable_quota_exhausted_http_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_quota_exhausted_http_error_response()],
                    "acc-2": [_text_claude_stream("第二个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    fill_priority=10,
                )
            )

            response, response_text = asyncio.run(
                _invoke_anthropic_messages_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 256,
                        "stream": False,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-2")
            self.assertIn("第二个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2"])
            attempt_logs = [
                item
                for item in _read_api_logs(settings)
                if item.get("phase") == "upstream_attempt"
            ]
            self.assertEqual(len(attempt_logs), 2)
            self.assertEqual(attempt_logs[0]["errorCode"], "429")
            self.assertEqual(
                [
                    (
                        item.get("event"),
                        item.get("attempt"),
                        item.get("accountId"),
                        item.get("success"),
                        item.get("stopReason"),
                    )
                    for item in attempt_logs
                ],
                [
                    ("v1_messages", 1, "acc-1", False, "upstream_turn_error"),
                    ("v1_messages", 2, "acc-2", True, "end_turn"),
                ],
            )

    def test_retryable_quota_exhausted_marks_account_unavailable_for_following_requests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [
                        _quota_exhausted_http_error_response(),
                        _quota_exhausted_http_error_response(),
                    ],
                    "acc-2": [
                        _text_claude_stream("第二个账号第一次命中"),
                        _text_claude_stream("第二个账号第二次命中"),
                    ],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    fill_priority=10,
                )
            )

            first_response, first_text = asyncio.run(
                _invoke_anthropic_messages_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 256,
                        "stream": False,
                        "messages": [{"role": "user", "content": "hello-1"}],
                    },
                )
            )
            second_response, second_text = asyncio.run(
                _invoke_anthropic_messages_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 256,
                        "stream": False,
                        "messages": [{"role": "user", "content": "hello-2"}],
                    },
                )
            )

            self.assertEqual(first_response.status_code, 200)
            self.assertEqual(first_response.headers["x-accio-account-id"], "acc-2")
            self.assertIn("第二个账号第一次命中", first_text)
            self.assertEqual(second_response.status_code, 200)
            self.assertEqual(second_response.headers["x-accio-account-id"], "acc-2")
            self.assertIn("第二个账号第二次命中", second_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2", "acc-2"])

            exhausted_account = store.get_account("acc-1")
            self.assertIsNotNone(exhausted_account)
            self.assertTrue(exhausted_account.auto_disabled)
            self.assertTrue(exhausted_account.manual_enabled)
            self.assertIsNotNone(exhausted_account.next_quota_check_at)
            self.assertEqual(
                exhausted_account.next_quota_check_reason,
                "上游 quota exhausted 后自动恢复重试",
            )
            self.assertIn(
                "quota exhausted",
                str(exhausted_account.auto_disabled_reason or "").lower(),
            )

    def test_retryable_quota_exhausted_walks_accounts_until_one_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_quota_exhausted_http_error_response()],
                    "acc-2": [_quota_exhausted_http_error_response()],
                    "acc-3": [_text_claude_stream("第三个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            for index in range(1, 4):
                store.save(
                    Account(
                        id=f"acc-{index}",
                        name=f"账号{index}",
                        access_token=f"token-{index}",
                        refresh_token=f"refresh-{index}",
                        utdid=f"utdid-{index}",
                        fill_priority=index,
                    )
                )

            response, response_text = asyncio.run(
                _invoke_anthropic_messages_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 256,
                        "stream": False,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-3")
            self.assertIn("第三个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2", "acc-3"])

            for exhausted_id in ("acc-1", "acc-2"):
                exhausted_account = store.get_account(exhausted_id)
                self.assertIsNotNone(exhausted_account)
                self.assertTrue(exhausted_account.auto_disabled)
                self.assertEqual(
                    exhausted_account.next_quota_check_reason,
                    "上游 quota exhausted 后自动恢复重试",
                )

    def test_openai_stream_quota_exhausted_walks_accounts_until_one_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_quota_exhausted_http_error_response()],
                    "acc-2": [_quota_exhausted_http_error_response()],
                    "acc-3": [_text_claude_stream("第三个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            for index in range(1, 4):
                store.save(
                    Account(
                        id=f"acc-{index}",
                        name=f"账号{index}",
                        access_token=f"token-{index}",
                        refresh_token=f"refresh-{index}",
                        utdid=f"utdid-{index}",
                        fill_priority=index,
                    )
                )

            response, response_text = asyncio.run(
                _invoke_openai_chat_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-3")
            self.assertIn("第三个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2", "acc-3"])

            for exhausted_id in ("acc-1", "acc-2"):
                exhausted_account = store.get_account(exhausted_id)
                self.assertIsNotNone(exhausted_account)
                self.assertTrue(exhausted_account.auto_disabled)
                self.assertEqual(
                    exhausted_account.next_quota_check_reason,
                    "上游 quota exhausted 后自动恢复重试",
                )

    def test_openai_non_stream_quota_exhausted_walks_accounts_until_one_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_quota_exhausted_http_error_response()],
                    "acc-2": [_quota_exhausted_http_error_response()],
                    "acc-3": [_text_claude_stream("第三个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            for index in range(1, 4):
                store.save(
                    Account(
                        id=f"acc-{index}",
                        name=f"账号{index}",
                        access_token=f"token-{index}",
                        refresh_token=f"refresh-{index}",
                        utdid=f"utdid-{index}",
                        fill_priority=index,
                    )
                )

            response, response_text = asyncio.run(
                _invoke_openai_chat_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "stream": False,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-3")
            self.assertIn("第三个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2", "acc-3"])

            for exhausted_id in ("acc-1", "acc-2"):
                exhausted_account = store.get_account(exhausted_id)
                self.assertIsNotNone(exhausted_account)
                self.assertTrue(exhausted_account.auto_disabled)
                self.assertEqual(
                    exhausted_account.next_quota_check_reason,
                    "上游 quota exhausted 后自动恢复重试",
                )

    def test_anthropic_stream_quota_exhausted_walks_accounts_until_one_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_quota_exhausted_http_error_response()],
                    "acc-2": [_quota_exhausted_http_error_response()],
                    "acc-3": [_text_claude_stream("第三个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            for index in range(1, 4):
                store.save(
                    Account(
                        id=f"acc-{index}",
                        name=f"账号{index}",
                        access_token=f"token-{index}",
                        refresh_token=f"refresh-{index}",
                        utdid=f"utdid-{index}",
                        fill_priority=index,
                    )
                )

            response, response_text = asyncio.run(
                _invoke_anthropic_messages_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 256,
                        "stream": True,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-3")
            self.assertIn("第三个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2", "acc-3"])

            for exhausted_id in ("acc-1", "acc-2"):
                exhausted_account = store.get_account(exhausted_id)
                self.assertIsNotNone(exhausted_account)
                self.assertTrue(exhausted_account.auto_disabled)
                self.assertEqual(
                    exhausted_account.next_quota_check_reason,
                    "上游 quota exhausted 后自动恢复重试",
                )

    def test_native_generate_content_quota_exhausted_walks_accounts_until_one_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_quota_exhausted_http_error_response()],
                    "acc-2": [_quota_exhausted_http_error_response()],
                    "acc-3": [_text_claude_stream("第三个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                    app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            for index in range(1, 4):
                store.save(
                    Account(
                        id=f"acc-{index}",
                        name=f"账号{index}",
                        access_token=f"token-{index}",
                        refresh_token=f"refresh-{index}",
                        utdid=f"utdid-{index}",
                        fill_priority=index,
                    )
                )

            response, response_text = asyncio.run(
                _invoke_native_generate_content_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-3")
            self.assertIn("第三个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2", "acc-3"])

            for exhausted_id in ("acc-1", "acc-2"):
                exhausted_account = store.get_account(exhausted_id)
                self.assertIsNotNone(exhausted_account)
                self.assertTrue(exhausted_account.auto_disabled)
                self.assertEqual(
                    exhausted_account.next_quota_check_reason,
                    "上游 quota exhausted 后自动恢复重试",
                )

    def test_gemini_generate_content_quota_exhausted_walks_accounts_until_one_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_quota_exhausted_http_error_response()],
                    "acc-2": [_quota_exhausted_http_error_response()],
                    "acc-3": [_gemini_text_response("第三个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            for index in range(1, 4):
                store.save(
                    Account(
                        id=f"acc-{index}",
                        name=f"账号{index}",
                        access_token=f"token-{index}",
                        refresh_token=f"refresh-{index}",
                        utdid=f"utdid-{index}",
                        fill_priority=index,
                    )
                )

            response, response_text = asyncio.run(
                _invoke_gemini_generate_content_route(
                    app,
                    headers={"x-goog-api-key": "admin"},
                    model_name="gemini-2.5-pro",
                    payload={
                        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-3")
            self.assertIn("第三个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2", "acc-3"])

            for exhausted_id in ("acc-1", "acc-2"):
                exhausted_account = store.get_account(exhausted_id)
                self.assertIsNotNone(exhausted_account)
                self.assertTrue(exhausted_account.auto_disabled)
                self.assertEqual(
                    exhausted_account.next_quota_check_reason,
                    "上游 quota exhausted 后自动恢复重试",
                )

    def test_native_generate_content_retries_once_after_retryable_upstream_turn_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [
                        _upstream_turn_error_stream(
                            code="555",
                            message="blocked by sentinel rate limit",
                        )
                    ],
                    "acc-2": [_text_claude_stream("第二个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                    app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    fill_priority=10,
                )
            )

            response, response_text = asyncio.run(
                _invoke_native_generate_content_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-accio-account-id"], "acc-2")
            self.assertIn("第二个账号命中", response_text)
            self.assertEqual(fake_client.calls, ["acc-1", "acc-2"])
            attempt_logs = [
                item
                for item in _read_api_logs(settings)
                if item.get("phase") == "upstream_attempt"
            ]
            self.assertEqual(len(attempt_logs), 2)
            self.assertEqual(
                [
                    (
                        item.get("event"),
                        item.get("attempt"),
                        item.get("accountId"),
                        item.get("success"),
                        item.get("stopReason"),
                    )
                    for item in attempt_logs
                ],
                [
                    ("native_generate_content", 1, "acc-1", False, "upstream_turn_error"),
                    ("native_generate_content", 2, "acc-2", True, "upstream_request_completed"),
                ],
            )

    def test_openai_responses_retry_logs_each_upstream_attempt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [
                        _upstream_turn_error_stream(
                            code="555",
                            message="blocked by sentinel rate limit",
                        )
                    ],
                    "acc-2": [_text_claude_stream("第二个账号命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )
            store.save(
                Account(
                    id="acc-2",
                    name="账号2",
                    access_token="token-2",
                    refresh_token="refresh-2",
                    utdid="utdid-2",
                    fill_priority=10,
                )
            )

            response, response_text = asyncio.run(
                _invoke_openai_responses_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "input": "hello",
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("第二个账号命中", response_text)
            attempt_logs = [
                item
                for item in _read_api_logs(settings)
                if item.get("phase") == "upstream_attempt"
            ]
            self.assertEqual(len(attempt_logs), 2)
            self.assertEqual(
                [
                    (
                        item.get("event"),
                        item.get("attempt"),
                        item.get("accountId"),
                        item.get("success"),
                        item.get("stopReason"),
                    )
                    for item in attempt_logs
                ],
                [
                    ("v1_responses", 1, "acc-1", False, "upstream_turn_error"),
                    ("v1_responses", 2, "acc-2", True, "end_turn"),
                ],
            )

    def test_gemini_generate_content_success_logs_upstream_attempt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_text_claude_stream("Gemini 兼容命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )

            with patch(
                "accio_panel.proxy_routes.context._decode_gemini_generate_content_response",
                return_value={
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": "Gemini 兼容命中"}],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 2,
                    },
                },
            ):
                response, response_text = asyncio.run(
                    _invoke_gemini_generate_content_route(
                        app,
                        headers={"x-goog-api-key": "admin"},
                        model_name="claude-sonnet-4-6",
                        payload={
                            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                        },
                    )
                )

            self.assertEqual(response.status_code, 200)
            self.assertIn("Gemini 兼容命中", response_text)
            attempt_logs = [
                item
                for item in _read_api_logs(settings)
                if item.get("phase") == "upstream_attempt"
            ]
            self.assertEqual(len(attempt_logs), 1)
            self.assertEqual(
                [
                    (
                        item.get("event"),
                        item.get("attempt"),
                        item.get("accountId"),
                        item.get("success"),
                        item.get("stopReason"),
                    )
                    for item in attempt_logs
                ],
                [
                    ("gemini_generate_content", 1, "acc-1", True, "STOP"),
                ],
            )

    def test_anthropic_messages_route_converts_to_reqtxt_standard_body(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_text_claude_stream("Anthropic 命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            app.state.store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )

            response, response_text = asyncio.run(
                _invoke_anthropic_messages_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 1024,
                        "stream": False,
                        "system": "system",
                        "messages": [{"role": "user", "content": "hello"}],
                        "tools": [
                            {
                                "name": "grep",
                                "description": "搜索代码",
                                "input_schema": {"type": "object", "properties": {}},
                            }
                        ],
                        "message_id": "msg-anthropic-1",
                        "session_key": "sess-anthropic-1",
                        "conversation_id": "conv-anthropic-1",
                        "conversation_name": "New conversation",
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("Anthropic 命中", response_text)
            self.assertEqual(len(fake_client.request_bodies), 1)
            upstream_body = fake_client.request_bodies[0]
            self.assertEqual(
                sorted(upstream_body.keys()),
                [
                    "contents",
                    "conversation_id",
                    "conversation_name",
                    "max_output_tokens",
                    "message_id",
                    "model",
                    "request_id",
                    "session_key",
                    "system_instruction",
                    "token",
                    "tools",
                ],
            )
            self.assertEqual(upstream_body["message_id"], "msg-anthropic-1")
            self.assertEqual(upstream_body["session_key"], "sess-anthropic-1")
            self.assertEqual(upstream_body["conversation_id"], "conv-anthropic-1")
            self.assertEqual(upstream_body["conversation_name"], "New conversation")
            self.assertEqual(upstream_body["tools"][0]["parameters_json"], '{"type": "object", "properties": {}}')
            self.assertNotIn("parametersJson", upstream_body["tools"][0])

    def test_openai_chat_route_converts_to_reqtxt_standard_body(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_text_claude_stream("OpenAI 命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            app.state.store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )

            response, response_text = asyncio.run(
                _invoke_openai_chat_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload={
                        "model": "claude-sonnet-4-6",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hello"}],
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "grep",
                                    "description": "搜索代码",
                                    "parameters": {"type": "object", "properties": {}},
                                },
                            }
                        ],
                        "tool_choice": "required",
                        "message_id": "msg-openai-1",
                        "session_id": "sess-openai-1",
                        "conversation_id": "conv-openai-1",
                        "conversation_name": "New conversation",
                    },
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("OpenAI 命中", response_text)
            self.assertEqual(len(fake_client.request_bodies), 1)
            upstream_body = fake_client.request_bodies[0]
            self.assertEqual(
                sorted(upstream_body.keys()),
                [
                    "contents",
                    "conversation_id",
                    "conversation_name",
                    "max_output_tokens",
                    "message_id",
                    "model",
                    "request_id",
                    "session_key",
                    "token",
                    "tools",
                ],
            )
            self.assertEqual(upstream_body["message_id"], "msg-openai-1")
            self.assertEqual(upstream_body["session_key"], "sess-openai-1")
            self.assertEqual(upstream_body["conversation_id"], "conv-openai-1")
            self.assertEqual(upstream_body["conversation_name"], "New conversation")
            self.assertEqual(upstream_body["tools"][0]["parameters_json"], '{"type": "object", "properties": {}}')
            self.assertNotIn("parametersJson", upstream_body["tools"][0])
            self.assertNotIn("tool_config", upstream_body)
            self.assertNotIn("properties", upstream_body)

    def test_native_generate_content_preserves_upstream_native_body_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_text_claude_stream("原生请求命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                    app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            store = app.state.store
            store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )

            request_payload = {
                "model": "claude-sonnet-4-6",
                "request_id": "req-native-1",
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "2025 年 1 月 20 日发生了什么大事？不允许联网"}],
                    }
                ],
                "system_instruction": "<identity>Accio</identity>",
                "max_output_tokens": 16384,
                "message_id": "msg-native-1",
                "session_key": "session-native-1",
                "conversation_id": "conversation-native-1",
                "conversation_name": "New conversation",
                "tools": [
                    {
                        "name": "grep",
                        "description": "搜索代码",
                        "parameters_json": json.dumps(
                            {
                                "type": "object",
                                "properties": {
                                    "pattern": {"type": "string"},
                                },
                                "required": ["pattern"],
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
            }

            response, response_text = asyncio.run(
                _invoke_native_generate_content_route(
                    app,
                    headers={"x-api-key": "admin"},
                    payload=request_payload,
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(fake_client.calls, ["acc-1"])
            self.assertIn("原生请求命中", response_text)

            self.assertEqual(len(fake_client.request_bodies), 1)
            upstream_body = fake_client.request_bodies[0]
            self.assertEqual(
                sorted(upstream_body.keys()),
                [
                    "contents",
                    "conversation_id",
                    "conversation_name",
                    "max_output_tokens",
                    "message_id",
                    "model",
                    "request_id",
                    "session_key",
                    "system_instruction",
                    "token",
                    "tools",
                ],
            )
            self.assertEqual(upstream_body["model"], "claude-sonnet-4-6")
            self.assertEqual(upstream_body["request_id"], "req-native-1")
            self.assertEqual(upstream_body["message_id"], "msg-native-1")
            self.assertEqual(upstream_body["system_instruction"], "<identity>Accio</identity>")
            self.assertEqual(upstream_body["max_output_tokens"], 16384)
            self.assertEqual(upstream_body["session_key"], "session-native-1")
            self.assertEqual(upstream_body["conversation_id"], "conversation-native-1")
            self.assertEqual(upstream_body["conversation_name"], "New conversation")
            self.assertEqual(upstream_body["token"], "token-1")
            self.assertEqual(
                upstream_body["contents"][0]["parts"][0]["text"],
                "2025 年 1 月 20 日发生了什么大事？不允许联网",
            )
            self.assertEqual(upstream_body["tools"][0]["name"], "grep")
            self.assertIn("parameters_json", upstream_body["tools"][0])
            self.assertNotIn("parametersJson", upstream_body["tools"][0])

    def test_gemini_generate_content_converts_to_reqtxt_standard_body(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(data_dir=Path(temp_dir), database_url="")
            fake_client = _FakeProxyClient(
                {
                    "acc-1": [_text_claude_stream("Gemini 兼容命中")],
                }
            )

            async def _noop_scheduler(_application):
                return None

            with patch("accio_panel.web.AccioClient", return_value=fake_client):
                with patch("accio_panel.proxy_routes.context._is_allowed_dynamic_model_impl", return_value=(True, [])):
                    with patch("accio_panel.web._quota_scheduler_loop", _noop_scheduler):
                        app = create_app(settings)

            app.state.panel_settings_store.save(
                PanelSettings(
                    admin_password="admin",
                    session_secret="test-session",
                    api_account_strategy="fill",
                )
            )
            app.state.store.save(
                Account(
                    id="acc-1",
                    name="账号1",
                    access_token="token-1",
                    refresh_token="refresh-1",
                    utdid="utdid-1",
                    fill_priority=0,
                )
            )

            selected_account = app.state.store.get_account("acc-1")
            self.assertIsNotNone(selected_account)
            with patch(
                "accio_panel.proxy_routes.context._select_proxy_account_impl",
                return_value=(selected_account, {"success": True, "remaining_value": None}),
            ), patch(
                "accio_panel.proxy_routes.context._decode_gemini_generate_content_response",
                return_value={
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": "Gemini 兼容命中"}],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 2,
                    },
                },
            ):
                response, response_text = asyncio.run(
                    _invoke_gemini_generate_content_route(
                        app,
                        headers={"x-goog-api-key": "admin"},
                        model_name="claude-sonnet-4-6",
                        payload={
                            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                            "system_instruction": "system",
                            "generationConfig": {"maxOutputTokens": 1024},
                            "tools": [
                                {
                                    "name": "grep",
                                    "description": "搜索代码",
                                    "parameters_json": '{"type":"object"}',
                                }
                            ],
                            "message_id": "msg-gemini-1",
                            "session_key": "sess-gemini-1",
                            "conversation_id": "conv-gemini-1",
                            "conversation_name": "New conversation",
                        },
                    )
                )

            self.assertEqual(response.status_code, 200)
            self.assertIn("Gemini 兼容命中", response_text)
            self.assertEqual(len(fake_client.request_bodies), 1)
            upstream_body = fake_client.request_bodies[0]
            self.assertEqual(
                sorted(upstream_body.keys()),
                [
                    "contents",
                    "conversation_id",
                    "conversation_name",
                    "max_output_tokens",
                    "message_id",
                    "model",
                    "request_id",
                    "session_key",
                    "system_instruction",
                    "token",
                    "tools",
                ],
            )
            self.assertEqual(upstream_body["model"], "claude-sonnet-4-6")
            self.assertEqual(upstream_body["max_output_tokens"], 1024)
            self.assertEqual(upstream_body["token"], "token-1")
            self.assertEqual(upstream_body["conversation_name"], "New conversation")
            self.assertEqual(upstream_body["tools"][0]["parameters_json"], '{"type":"object"}')
            self.assertNotIn("parametersJson", upstream_body["tools"][0])


if __name__ == "__main__":
    unittest.main()
