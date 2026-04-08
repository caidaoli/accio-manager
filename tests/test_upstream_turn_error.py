import json
import unittest

from accio_panel.anthropic_proxy import UpstreamTurnError, iter_anthropic_sse_events


class _FakeSSEUpstreamResponse:
    def __init__(self, payloads: list[dict[str, object]]):
        self._payloads = list(payloads)

    def iter_lines(self, decode_unicode: bool = False):
        for payload in self._payloads:
            line = f"data: {json.dumps(payload, ensure_ascii=False)}"
            if decode_unicode:
                yield line
            else:
                yield line.encode("utf-8")

    def close(self):
        return None


class UpstreamTurnErrorTests(unittest.TestCase):
    def test_iter_anthropic_sse_events_raises_on_turn_complete_error(self):
        response = _FakeSSEUpstreamResponse(
            [
                {
                    "turn_complete": True,
                    "error_code": "505",
                    "error_message": "internal server error",
                }
            ]
        )

        with self.assertRaises(UpstreamTurnError) as context:
            list(iter_anthropic_sse_events(response, "claude-sonnet-4-6"))

        self.assertEqual(context.exception.error_code, "505")
        self.assertEqual(context.exception.error_message, "internal server error")


if __name__ == "__main__":
    unittest.main()
