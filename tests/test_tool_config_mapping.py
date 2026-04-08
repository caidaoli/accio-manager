import json
import unittest

from accio_panel.anthropic_proxy import build_accio_request
from accio_panel.gemini_proxy import build_generate_content_request
from accio_panel.openai_proxy import build_accio_request_from_openai


REQTXT_KEYS = [
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
]

LEGACY_KEYS = [
    "utdid",
    "version",
    "empid",
    "tenant",
    "iai_tag",
    "stream",
    "incremental",
    "stop_sequences",
    "properties",
    "tool_config",
    "temperature",
    "top_p",
    "response_format",
    "include_thoughts",
    "thinking_level",
    "thinking_budget",
]


class RequestShapeTests(unittest.TestCase):
    def assert_reqtxt_shape(self, request_body):
        self.assertEqual(sorted(request_body.keys()), REQTXT_KEYS)
        for legacy_key in LEGACY_KEYS:
            self.assertNotIn(legacy_key, request_body)

    def test_anthropic_builder_matches_reqtxt_shape(self):
        request_body = build_accio_request(
            {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hello"}],
                "system": "system",
                "tools": [
                    {
                        "name": "search_docs",
                        "description": "搜索文档",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
                "session_key": "sess-1",
                "conversation_id": "conv-1",
                "conversation_name": "New conversation",
            },
            token="token-1",
            utdid="utdid-1",
            version="1.0.0",
        )

        self.assert_reqtxt_shape(request_body)
        self.assertEqual(request_body["system_instruction"], "system")
        self.assertEqual(request_body["tools"][0]["parameters_json"], '{"type": "object", "properties": {}}')
        self.assertNotIn("parametersJson", request_body["tools"][0])

    def test_openai_builder_matches_reqtxt_shape(self):
        request_body = build_accio_request_from_openai(
            {
                "model": "gpt-5.4",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_docs",
                            "description": "搜索文档",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "search_docs"},
                },
                "session_id": "sess-1",
                "conversation_id": "conv-1",
                "conversation_name": "New conversation",
                "previous_response_id": "resp-1",
                "metadata": {"ignored": True},
            },
            token="token-1",
            utdid="utdid-1",
            version="1.0.0",
        )

        self.assert_reqtxt_shape(request_body)
        self.assertEqual(request_body["session_key"], "sess-1")
        self.assertEqual(request_body["conversation_id"], "conv-1")
        self.assertEqual(request_body["conversation_name"], "New conversation")
        self.assertEqual(
            request_body["tools"][0]["parameters_json"],
            json.dumps(
                {
                    "type": "object",
                    "properties": {},
                },
                ensure_ascii=False,
            ),
        )
        self.assertNotIn("parametersJson", request_body["tools"][0])

    def test_generate_content_request_matches_reqtxt_shape(self):
        request_body = build_generate_content_request(
            {
                "model": "gemini-3-flash-preview",
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                "system_instruction": "system",
                "tools": [
                    {
                        "name": "search_docs",
                        "description": "搜索文档",
                        "parameters_json": '{"type":"object"}',
                    }
                ],
                "max_output_tokens": 1024,
                "session_key": "sess-1",
                "conversation_id": "conv-1",
                "conversation_name": "New conversation",
            },
            token="token-1",
        )

        self.assert_reqtxt_shape(request_body)
        self.assertEqual(request_body["tools"][0]["parameters_json"], '{"type":"object"}')
        self.assertNotIn("parametersJson", request_body["tools"][0])

    def test_openai_tool_choice_does_not_emit_unsupported_fields(self):
        for tool_choice in (
            "auto",
            "none",
            "required",
            {"type": "function", "function": {"name": "search_docs"}},
        ):
            with self.subTest(tool_choice=tool_choice):
                request_body = build_accio_request_from_openai(
                    {
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "hello"}],
                        "tool_choice": tool_choice,
                    },
                    token="token-1",
                    utdid="utdid-1",
                    version="1.0.0",
                )

                self.assertNotIn("tool_config", request_body)
                self.assertNotIn("properties", request_body)


if __name__ == "__main__":
    unittest.main()
