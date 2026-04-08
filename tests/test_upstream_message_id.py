import unittest

from accio_panel.anthropic_proxy import build_accio_request
from accio_panel.gemini_proxy import build_generate_content_request
from accio_panel.openai_proxy import build_accio_request_from_openai


class UpstreamMessageIdTests(unittest.TestCase):
    def test_builders_generate_message_id_when_missing(self):
        scenarios = {
            "anthropic": lambda: build_accio_request(
                {
                    "model": "claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                token="token-1",
                utdid="utdid-1",
                version="1.0.0",
            ),
            "openai": lambda: build_accio_request_from_openai(
                {
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                token="token-1",
                utdid="utdid-1",
                version="1.0.0",
            ),
            "generate_content": lambda: build_generate_content_request(
                {
                    "model": "gemini-3-flash-preview",
                    "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                },
                token="token-1",
            ),
        }

        for name, builder in scenarios.items():
            with self.subTest(builder=name):
                request_body = builder()
                self.assertIsInstance(request_body.get("message_id"), str)
                self.assertTrue(request_body["message_id"])

    def test_builders_preserve_explicit_message_id(self):
        scenarios = {
            "anthropic": lambda message_id: build_accio_request(
                {
                    "model": "claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": "hello"}],
                    "message_id": message_id,
                },
                token="token-1",
                utdid="utdid-1",
                version="1.0.0",
            ),
            "openai": lambda message_id: build_accio_request_from_openai(
                {
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "hello"}],
                    "message_id": message_id,
                },
                token="token-1",
                utdid="utdid-1",
                version="1.0.0",
            ),
            "generate_content": lambda message_id: build_generate_content_request(
                {
                    "model": "gemini-3-flash-preview",
                    "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                    "message_id": message_id,
                },
                token="token-1",
            ),
        }

        for name, builder in scenarios.items():
            with self.subTest(builder=name):
                self.assertEqual(builder("msg-fixed-id")["message_id"], "msg-fixed-id")


if __name__ == "__main__":
    unittest.main()
