"""Tests for the HTTP interface.

Each test class starts a real server on an ephemeral port and talks to it over a
real socket with :mod:`urllib`. That is slower than calling the handler directly,
but it is the only way to catch the mistakes that actually bite here: a wrong
Content-Length, a body left unread on a keep-alive connection, a status code that
does not match the cache's answer.
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from typing import Any

from lrucache import LRUCache
from lrucache.server import MAX_BODY_BYTES, create_server

from .test_ttl import FakeClock


class Response:
    """The parts of an HTTP response these tests care about."""

    def __init__(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        self.status = status
        self.body = body
        self.headers = headers

    def json(self) -> Any:
        return json.loads(self.body)


def request(
    url: str, method: str = "GET", payload: Any = None, raw_body: bytes | None = None
) -> Response:
    """Make one HTTP request, returning the response even when it is an error."""
    body = raw_body
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    if body is not None:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return Response(response.status, response.read(), dict(response.headers))
    except urllib.error.HTTPError as exc:
        return Response(exc.code, exc.read(), dict(exc.headers))


class ServerTestCase(unittest.TestCase):
    """Starts a quiet server on a free port for the duration of each test."""

    capacity = 8
    default_ttl: float | None = None

    def setUp(self) -> None:
        self.clock = FakeClock()
        self.cache: LRUCache[str, Any] = LRUCache(
            self.capacity, default_ttl=self.default_ttl, clock=self.clock
        )
        self.server = create_server(self.cache, host="127.0.0.1", port=0, log_requests=False)
        # A short poll interval so shutdown() returns promptly; the default 0.5s
        # would otherwise dominate the runtime of this whole suite.
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()
        self.base = self.server.url

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.cache._check_invariants()

    def key_url(self, key: str) -> str:
        return f"{self.base}/cache/{key}"


class GetTests(ServerTestCase):
    def test_get_returns_a_cached_value(self) -> None:
        self.cache.put("greeting", "hello")

        response = request(self.key_url("greeting"))

        self.assertEqual(response.status, 200)
        self.assertEqual(response.json(), {"key": "greeting", "value": "hello"})
        self.assertEqual(response.headers["Content-Type"], "application/json")

    def test_get_on_a_missing_key_is_404(self) -> None:
        response = request(self.key_url("absent"))

        self.assertEqual(response.status, 404)
        self.assertIn("detail", response.json())

    def test_get_on_an_expired_key_is_404(self) -> None:
        self.cache.put("temp", "value", ttl=5)
        self.clock.advance(10)

        self.assertEqual(request(self.key_url("temp")).status, 404)

    def test_get_counts_as_a_cache_hit(self) -> None:
        self.cache.put("a", 1)

        request(self.key_url("a"))

        self.assertEqual(self.cache.stats().hits, 1)

    def test_get_promotes_the_entry(self) -> None:
        cache: LRUCache[str, Any] = self.cache
        for key in ("a", "b", "c"):
            cache.put(key, 1)

        request(self.key_url("a"))

        self.assertEqual(cache.keys()[0], "a")

    def test_json_value_types_round_trip(self) -> None:
        payloads: dict[str, Any] = {
            "number": 42,
            "float": 1.5,
            "string": "text",
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "object": {"nested": {"deep": True}},
        }
        for key, value in payloads.items():
            with self.subTest(key=key):
                self.assertEqual(request(self.key_url(key), "PUT", {"value": value}).status, 204)
                self.assertEqual(request(self.key_url(key)).json()["value"], value)

    def test_unknown_path_is_404(self) -> None:
        self.assertEqual(request(f"{self.base}/nope").status, 404)

    def test_cache_root_without_a_key_is_404(self) -> None:
        self.assertEqual(request(f"{self.base}/cache/").status, 404)


class PutTests(ServerTestCase):
    def test_put_stores_a_value_and_returns_204(self) -> None:
        response = request(self.key_url("a"), "PUT", {"value": "stored"})

        self.assertEqual(response.status, 204)
        self.assertEqual(response.body, b"")
        self.assertEqual(self.cache.peek("a"), "stored")

    def test_put_overwrites_an_existing_key(self) -> None:
        request(self.key_url("a"), "PUT", {"value": "first"})
        request(self.key_url("a"), "PUT", {"value": "second"})

        self.assertEqual(self.cache.peek("a"), "second")
        self.assertEqual(len(self.cache), 1)

    def test_put_accepts_a_ttl(self) -> None:
        request(self.key_url("a"), "PUT", {"value": 1, "ttl": 10})
        self.clock.advance(10)

        self.assertEqual(request(self.key_url("a")).status, 404)

    def test_put_with_a_null_ttl_stores_without_expiry(self) -> None:
        request(self.key_url("a"), "PUT", {"value": 1, "ttl": None})
        self.clock.advance(10_000)

        self.assertEqual(request(self.key_url("a")).status, 200)

    def test_put_can_store_a_null_value(self) -> None:
        """A stored null must come back as a 200, not be confused with a miss."""
        request(self.key_url("a"), "PUT", {"value": None})

        response = request(self.key_url("a"))

        self.assertEqual(response.status, 200)
        self.assertEqual(response.json(), {"key": "a", "value": None})

    def test_put_past_capacity_evicts(self) -> None:
        cache: LRUCache[str, Any] = LRUCache(2)
        server = create_server(cache, host="127.0.0.1", port=0, log_requests=False)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        try:
            for key in ("a", "b", "c"):
                request(f"{server.url}/cache/{key}", "PUT", {"value": 1})

            self.assertEqual(request(f"{server.url}/cache/a").status, 404)
            self.assertEqual(len(cache), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_body_without_a_value_field_is_400(self) -> None:
        response = request(self.key_url("a"), "PUT", {"ttl": 10})

        self.assertEqual(response.status, 400)
        self.assertIn("value", response.json()["detail"])

    def test_malformed_json_body_is_400(self) -> None:
        response = request(self.key_url("a"), "PUT", raw_body=b"{not json")

        self.assertEqual(response.status, 400)
        self.assertEqual(len(self.cache), 0)

    def test_non_object_json_body_is_400(self) -> None:
        response = request(self.key_url("a"), "PUT", raw_body=b"[1, 2, 3]")

        self.assertEqual(response.status, 400)
        self.assertIn("object", response.json()["detail"])

    def test_empty_body_is_400(self) -> None:
        response = request(self.key_url("a"), "PUT", raw_body=b"")

        self.assertEqual(response.status, 400)

    def test_non_numeric_ttl_is_400(self) -> None:
        response = request(self.key_url("a"), "PUT", {"value": 1, "ttl": "soon"})

        self.assertEqual(response.status, 400)
        self.assertIn("ttl", response.json()["detail"])

    def test_non_positive_ttl_is_400(self) -> None:
        """The cache's own validation error becomes a 400, not a 500."""
        response = request(self.key_url("a"), "PUT", {"value": 1, "ttl": 0})

        self.assertEqual(response.status, 400)
        self.assertEqual(len(self.cache), 0)

    def test_oversized_body_is_413(self) -> None:
        oversized = json.dumps({"value": "x" * (MAX_BODY_BYTES + 100)}).encode("utf-8")

        response = request(self.key_url("a"), "PUT", raw_body=oversized)

        self.assertEqual(response.status, 413)
        self.assertEqual(len(self.cache), 0)


class DeleteTests(ServerTestCase):
    def test_delete_removes_a_key_and_returns_204(self) -> None:
        self.cache.put("a", 1)

        response = request(self.key_url("a"), "DELETE")

        self.assertEqual(response.status, 204)
        self.assertNotIn("a", self.cache)

    def test_delete_on_a_missing_key_is_404(self) -> None:
        self.assertEqual(request(self.key_url("absent"), "DELETE").status, 404)

    def test_delete_on_an_expired_key_is_404(self) -> None:
        self.cache.put("a", 1, ttl=5)
        self.clock.advance(10)

        self.assertEqual(request(self.key_url("a"), "DELETE").status, 404)

    def test_delete_is_not_idempotent_in_its_status_code(self) -> None:
        """Second delete reports 404, which is how a client learns it was already gone."""
        self.cache.put("a", 1)

        self.assertEqual(request(self.key_url("a"), "DELETE").status, 204)
        self.assertEqual(request(self.key_url("a"), "DELETE").status, 404)


class StatsTests(ServerTestCase):
    def test_stats_reports_the_counters(self) -> None:
        self.cache.put("a", 1)
        self.cache.get("a")
        self.cache.get("absent")

        stats = request(f"{self.base}/stats").json()

        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)
        self.assertEqual(stats["capacity"], self.capacity)
        self.assertEqual(stats["size"], 1)

    def test_stats_on_an_untouched_cache_is_all_zeroes(self) -> None:
        stats = request(f"{self.base}/stats").json()

        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["hit_rate"], 0.0)

    def test_stats_reflects_requests_made_over_http(self) -> None:
        request(self.key_url("a"), "PUT", {"value": 1})
        request(self.key_url("a"))
        request(self.key_url("absent"))

        stats = request(f"{self.base}/stats").json()

        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["evictions"], 0)


class KeyEncodingTests(ServerTestCase):
    def test_percent_encoded_key_is_decoded(self) -> None:
        request(f"{self.base}/cache/a%2Fb", "PUT", {"value": 1})

        self.assertIn("a/b", self.cache)
        self.assertEqual(request(f"{self.base}/cache/a%2Fb").json()["value"], 1)

    def test_key_with_a_literal_slash_works(self) -> None:
        request(f"{self.base}/cache/nested/path", "PUT", {"value": 1})

        self.assertIn("nested/path", self.cache)

    def test_key_with_spaces_and_unicode(self) -> None:
        request(f"{self.base}/cache/caf%C3%A9%20au%20lait", "PUT", {"value": 1})

        self.assertIn("café au lait", self.cache)

    def test_query_string_is_not_part_of_the_key(self) -> None:
        request(f"{self.base}/cache/a?ignored=1", "PUT", {"value": 1})

        self.assertIn("a", self.cache)
        self.assertNotIn("a?ignored=1", self.cache)


class MethodTests(ServerTestCase):
    def test_post_on_a_key_is_405_with_an_allow_header(self) -> None:
        response = request(self.key_url("a"), "POST", {"value": 1})

        self.assertEqual(response.status, 405)
        self.assertIn("PUT", response.headers["Allow"])

    def test_put_on_stats_is_405(self) -> None:
        response = request(f"{self.base}/stats", "PUT", {"value": 1})

        self.assertEqual(response.status, 405)
        self.assertEqual(response.headers["Allow"], "GET, OPTIONS")

    def test_post_on_an_unknown_path_is_404(self) -> None:
        self.assertEqual(request(f"{self.base}/nope", "POST", {"value": 1}).status, 404)

    def test_options_advertises_the_allowed_methods(self) -> None:
        response = request(self.key_url("a"), "OPTIONS")

        self.assertEqual(response.status, 204)
        self.assertIn("DELETE", response.headers["Allow"])


class ConnectionReuseTests(ServerTestCase):
    def test_many_sequential_requests_on_one_opener(self) -> None:
        """Catches Content-Length bugs: a wrong one desynchronises the next request."""
        opener = urllib.request.build_opener()

        for index in range(25):
            url = self.key_url(f"key-{index}")
            put = urllib.request.Request(
                url, data=json.dumps({"value": index}).encode("utf-8"), method="PUT"
            )
            with opener.open(put, timeout=10) as response:
                self.assertEqual(response.status, 204)

            with opener.open(url, timeout=10) as response:
                self.assertEqual(json.loads(response.read())["value"], index)

    def test_a_rejected_request_does_not_break_the_next_one(self) -> None:
        request(self.key_url("a"), "PUT", raw_body=b"{bad json")

        response = request(self.key_url("a"), "PUT", {"value": "fine"})

        self.assertEqual(response.status, 204)
        self.assertEqual(self.cache.peek("a"), "fine")


class ConcurrentRequestTests(ServerTestCase):
    capacity = 16

    def test_concurrent_clients_leave_the_cache_consistent(self) -> None:
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                for op in range(20):
                    key = f"key-{op % 30}"
                    request(self.key_url(key), "PUT", {"value": index})
                    request(self.key_url(key))
            except BaseException as exc:  # noqa: BLE001 - recorded and re-raised below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        self.assertEqual(errors, [])
        self.assertLessEqual(len(self.cache), self.capacity)
        self.cache._check_invariants()


class DefaultTtlServerTests(ServerTestCase):
    default_ttl = 10

    def test_put_without_a_ttl_inherits_the_server_default(self) -> None:
        request(self.key_url("a"), "PUT", {"value": 1})
        self.clock.advance(10)

        self.assertEqual(request(self.key_url("a")).status, 404)

    def test_an_explicit_ttl_overrides_the_default(self) -> None:
        request(self.key_url("a"), "PUT", {"value": 1, "ttl": 100})
        self.clock.advance(10)

        self.assertEqual(request(self.key_url("a")).status, 200)


if __name__ == "__main__":
    unittest.main()
