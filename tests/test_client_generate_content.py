import unittest
from unittest.mock import Mock

from accio_panel.client import AccioClient
from accio_panel.config import Settings
from accio_panel.models import Account


class ClientGenerateContentTests(unittest.TestCase):
    def test_generate_content_uses_reqtxt_headers(self):
        client = AccioClient(Settings())
        client._session.post = Mock()
        account = Account(
            id="acc-1",
            name="账号1",
            access_token="token-1",
            refresh_token="refresh-1",
            utdid="utdid-1",
        )

        client.generate_content(account, {"model": "claude-sonnet-4-6"})

        _, kwargs = client._session.post.call_args
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(kwargs["headers"]["Accept"], "text/event-stream")
        self.assertEqual(kwargs["headers"]["utdid"], "utdid-1")
        self.assertEqual(kwargs["headers"]["version"], Settings().version)
        self.assertEqual(kwargs["headers"]["appKey"], "35298846")


if __name__ == "__main__":
    unittest.main()
