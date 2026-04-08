import unittest

from accio_panel.gemini_proxy import build_accio_request_from_gemini
from accio_panel.openai_proxy import build_accio_request_from_openai


class ToolConfigMappingTests(unittest.TestCase):
    def test_openai_string_tool_choice_maps_to_function_calling_modes(self):
        expectations = {
            "auto": {"functionCallingConfig": {"mode": "AUTO"}},
            "none": {"functionCallingConfig": {"mode": "NONE"}},
            "required": {"functionCallingConfig": {"mode": "ANY"}},
        }

        for tool_choice, expected_tool_config in expectations.items():
            with self.subTest(tool_choice=tool_choice):
                request_body = build_accio_request_from_openai(
                    {
                        "model": "deepseek-r1",
                        "messages": [{"role": "user", "content": "hello"}],
                        "tool_choice": tool_choice,
                    },
                    token="token-1",
                    utdid="utdid-1",
                    version="1.0.0",
                )

                self.assertEqual(request_body.get("tool_config"), expected_tool_config)

    def test_openai_named_tool_choice_maps_to_top_level_tool_config(self):
        request_body = build_accio_request_from_openai(
            {
                "model": "deepseek-r1",
                "messages": [{"role": "user", "content": "hello"}],
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
            },
            token="token-1",
            utdid="utdid-1",
            version="1.0.0",
        )

        self.assertEqual(
            request_body.get("tool_config"),
            {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": ["search_docs"],
                }
            },
        )
        self.assertNotIn("openai_tool_choice", request_body.get("properties", {}))

    def test_gemini_tool_config_stays_at_top_level_body(self):
        tool_config = {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": ["search_docs"],
            }
        }

        request_body = build_accio_request_from_gemini(
            {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                "tool_config": tool_config,
            },
            model="gemini-3-flash-preview",
            token="token-1",
            utdid="utdid-1",
            version="1.0.0",
        )

        self.assertEqual(request_body.get("tool_config"), tool_config)
        self.assertNotIn("tool_config", request_body.get("properties", {}))


if __name__ == "__main__":
    unittest.main()
