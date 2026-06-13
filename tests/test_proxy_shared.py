import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from accio_panel.api_logs import ApiLogStore
from accio_panel.app_settings import PanelSettings
from accio_panel.models import Account
from accio_panel.proxy_routes.shared import ProxyEndpointConfig, make_build_stream_attempt
from accio_panel.usage_stats import UsageStatsStore


def _read_logged_entries(log_file: Path) -> list[dict[str, Any]]:
    if not log_file.exists():
        return []
    return [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ProxySharedLogTests(unittest.TestCase):
    def test_stream_final_log_is_explicitly_final_and_not_an_upstream_attempt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            api_log_store = ApiLogStore(temp_path / "api.log")
            usage_stats_store = UsageStatsStore(temp_path / "stats.json")
            account = Account(
                id="acc-1",
                name="账号1",
                access_token="token-1",
                refresh_token="refresh-1",
                utdid="utdid-1",
            )
            config = ProxyEndpointConfig(
                event="v1_messages",
                model="claude-opus-4-8",
                default_stop_reason="end_turn",
                stream_complete_message="Anthropic 上游流式请求完成",
                error_response_builder=lambda *_args: None,
                extra_fields_extractor=lambda summary: {
                    "textChars": int(summary.get("text_chars") or 0),
                    "toolUseBlocks": int(summary.get("tool_use_blocks") or 0),
                },
            )

            def record_attempt(*_args: Any, **_kwargs: Any) -> None:
                return None

            def iter_sse_bytes(
                _response: object,
                _model: str,
                *,
                on_complete: Any,
            ):
                on_complete(
                    {
                        "usage": {"input_tokens": 539, "output_tokens": 30},
                        "stop_reason": "end_turn",
                        "text_chars": 35,
                        "tool_use_blocks": 0,
                    }
                )
                return iter([b"data: ok\n\n"])

            build_stream_attempt = make_build_stream_attempt(
                config=config,
                panel_settings=PanelSettings(api_account_strategy="fill"),
                store=object(),
                usage_stats_store=usage_stats_store,
                api_log_store=api_log_store,
                started_at=0.0,
                messages_count=1,
                record_attempt=record_attempt,
                disable_account_model_on_empty_response=lambda *_args, **_kwargs: None,
                clear_account_sentinel_rate_limit=lambda *_args, **_kwargs: None,
                empty_response_log_message=lambda *_args, **_kwargs: "空回复",
                iter_sse_bytes=iter_sse_bytes,
                chunk_has_meaningful_output=lambda _chunk: True,
            )

            stream_bytes, has_meaningful_output = build_stream_attempt(
                account,
                {"remaining_value": 20, "used_value": 0, "success": True},
                object(),
                "req-1",
                1,
                0.0,
            )

            self.assertTrue(has_meaningful_output)
            self.assertEqual(list(stream_bytes), [b"data: ok\n\n"])
            entries = _read_logged_entries(api_log_store.file_path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["phase"], "final")
            self.assertEqual(entries[0]["attempt"], 0)
            self.assertEqual(entries[0]["rootRequestId"], "req-1")
            self.assertEqual(entries[0]["message"], "流式调用完成")
            self.assertNotIn("上游", entries[0]["message"])


if __name__ == "__main__":
    unittest.main()
