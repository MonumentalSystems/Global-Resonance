import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from backend import server as api_server


class _SolarUpstream(BaseHTTPRequestHandler):
    health_status = 200

    def do_GET(self):
        if self.path == "/api/solar/health":
            body = json.dumps(
                {
                    "status": "ok" if self.health_status == 200 else "stale",
                    "alerting_ready": self.health_status == 200,
                }
            ).encode()
            self.send_response(self.health_status)
            self.send_header("Content-Type", "application/json")
        elif self.path == "/api/solar/metrics":
            body = (
                'event: metrics\n'
                'data: {"data_quality":{"alerting_ready":false}}\n\n'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
        else:
            body = b'{}'
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


class SolarProxyContractTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _SolarUpstream)
        cls.thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.thread.start()
        cls.previous_url = api_server.SOLAR_MONITOR_URL
        host, port = cls.upstream.server_address
        api_server.SOLAR_MONITOR_URL = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        api_server.SOLAR_MONITOR_URL = cls.previous_url
        cls.upstream.shutdown()
        cls.upstream.server_close()
        cls.thread.join(timeout=2)

    async def test_health_preserves_upstream_200_and_503(self):
        _SolarUpstream.health_status = 200
        response = await api_server._solar_proxy("health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(json.loads(response.body)["alerting_ready"])

        _SolarUpstream.health_status = 503
        response = await api_server._solar_proxy("health")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(json.loads(response.body)["alerting_ready"])

    async def test_named_metrics_event_framing_survives_proxy(self):
        response = await api_server._sse_proxy("metrics")
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        stream = "".join(chunks)
        self.assertIn("event: metrics\n", stream)
        self.assertIn('data: {"data_quality":{"alerting_ready":false}}\n\n', stream)


if __name__ == "__main__":
    unittest.main()
