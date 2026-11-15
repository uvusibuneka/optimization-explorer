import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from tools import openkilo_gateway as gateway


class KiloForwardTests(unittest.TestCase):
    def test_forwards_auto_free_without_authorization(self):
        class Response:
            status = 200
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self): return b'{"choices":[]}'

        with patch.object(gateway.urllib.request, "urlopen", return_value=Response()) as urlopen:
            result = gateway.forward_completion({
                "model": "openai/worker", "messages": [{"role": "user", "content": "x"}],
                "temperature": 0, "timeout": 10, "max_retries": 3,
            })
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(result, {"choices": []})
        self.assertEqual(payload["model"], "kilo-auto/free")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["temperature"], 0)
        self.assertNotIn("timeout", payload)
        self.assertIsNone(request.get_header("Authorization"))

    def test_preserves_upstream_429_body_and_retry_metadata(self):
        error = HTTPError(
            gateway.UPSTREAM, 429, "Too Many Requests",
            {"Retry-After": "7"}, io.BytesIO(b'{"error_type":"rate_limit_exceeded"}'))
        with patch.object(gateway.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(gateway.UpstreamError) as caught:
                gateway.forward_completion({"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(json.loads(caught.exception.body), {"error_type": "rate_limit_exceeded"})
        self.assertEqual(caught.exception.headers.get("Retry-After"), "7")

    def test_requires_messages_array(self):
        with self.assertRaisesRegex(ValueError, "messages array"):
            gateway.forward_completion({"model": "worker"})


if __name__ == "__main__":
    unittest.main()
