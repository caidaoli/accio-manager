import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import requests

from accio_panel.api_logs import ApiLogStore
from accio_panel.models import Account
from accio_panel.upstream_support import (
    make_upstream_attempt_logger,
    request_upstream_or_error,
)


class _FakeErrorResponse:
    def __init__(self, status_code: int, message: str, stop_reason: str):
        self.status_code = status_code
        self.message = message
        self.stop_reason = stop_reason


class _FakeUpstreamResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text
        self.ok = status_code < 400
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _read_logged_entries(log_file: Path) -> list[dict[str, object]]:
    if not log_file.exists():
        return []
    return [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class UpstreamSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.account = Account(
            id="acc-1",
            name="账号1",
            access_token="token-1",
            refresh_token="refresh-1",
            utdid="utdid-1",
            fill_priority=3,
        )
        self.quota = {"remaining_value": 77, "used_value": 23}

    def test_request_upstream_or_error_records_request_exception(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "api.log"
            store = ApiLogStore(log_file)
            record_attempt = make_upstream_attempt_logger(
                store,
                event="v1_chat_completions",
                model="claude-sonnet-4-6",
                strategy="fill",
                root_request_id="root-request",
                messages_count=2,
                max_tokens=512,
            )

            async def _raise_request_exception():
                raise requests.RequestException("boom")

            result = asyncio.run(
                request_upstream_or_error(
                    _raise_request_exception,
                    account=self.account,
                    quota=self.quota,
                    request_id="attempt-request",
                    attempt=2,
                    stream=True,
                    started_at=0.0,
                    record_attempt=record_attempt,
                    build_error_response=lambda status, message, stop_reason: _FakeErrorResponse(
                        status,
                        message,
                        stop_reason,
                    ),
                    retry_reason="empty_response",
                )
            )

            self.assertEqual(result.status_code, 502)
            self.assertEqual(result.stop_reason, "request_exception")
            entries = _read_logged_entries(log_file)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["phase"], "upstream_attempt")
            self.assertEqual(entries[0]["attempt"], 2)
            self.assertEqual(entries[0]["rootRequestId"], "root-request")
            self.assertEqual(entries[0]["retryReason"], "empty_response")
            self.assertEqual(entries[0]["stopReason"], "request_exception")

    def test_request_upstream_or_error_records_upstream_error_and_closes_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "api.log"
            store = ApiLogStore(log_file)
            record_attempt = make_upstream_attempt_logger(
                store,
                event="v1_messages",
                model="claude-sonnet-4-6",
                strategy="fill",
                root_request_id="root-request",
            )
            upstream_response = _FakeUpstreamResponse(503, "upstream failed")

            async def _return_error_response():
                return upstream_response

            result = asyncio.run(
                request_upstream_or_error(
                    _return_error_response,
                    account=self.account,
                    quota=self.quota,
                    request_id="attempt-request",
                    attempt=1,
                    stream=False,
                    started_at=0.0,
                    record_attempt=record_attempt,
                    build_error_response=lambda status, message, stop_reason: _FakeErrorResponse(
                        status,
                        message,
                        stop_reason,
                    ),
                )
            )

            self.assertEqual(result.status_code, 503)
            self.assertEqual(result.message, "upstream failed")
            self.assertEqual(result.stop_reason, "upstream_error")
            self.assertTrue(upstream_response.closed)
            entries = _read_logged_entries(log_file)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["statusCode"], 503)
            self.assertEqual(entries[0]["message"], "upstream failed")
            self.assertEqual(entries[0]["stopReason"], "upstream_error")

    def test_request_upstream_or_error_returns_success_response_without_logging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "api.log"
            store = ApiLogStore(log_file)
            record_attempt = make_upstream_attempt_logger(
                store,
                event="native_generate_content",
                model="gemini-2.5-pro",
                strategy="fill",
                root_request_id="root-request",
            )
            upstream_response = _FakeUpstreamResponse(200, "ok")

            async def _return_success_response():
                return upstream_response

            result = asyncio.run(
                request_upstream_or_error(
                    _return_success_response,
                    account=self.account,
                    quota=self.quota,
                    request_id="attempt-request",
                    attempt=1,
                    stream=True,
                    started_at=0.0,
                    record_attempt=record_attempt,
                    build_error_response=lambda status, message, stop_reason: _FakeErrorResponse(
                        status,
                        message,
                        stop_reason,
                    ),
                )
            )

            self.assertIs(result, upstream_response)
            self.assertEqual(_read_logged_entries(log_file), [])


if __name__ == "__main__":
    unittest.main()
