"""A small JSON/HTTP interface to an :class:`~lrucache.LRUCache`.

Built on :mod:`http.server` so the project keeps its zero-dependency promise.
``ThreadingHTTPServer`` handles each request on its own thread, which means the
cache's locking is genuinely exercised rather than merely present.

Routes
------

====== ================ ==============================================
GET    ``/cache/<key>``  200 with the value, or 404 if absent/expired
PUT    ``/cache/<key>``  204; body is ``{"value": ..., "ttl": ...}``
DELETE ``/cache/<key>``  204 if removed, 404 if it was not there
GET    ``/stats``        200 with the counters and hit rate
====== ================ ==============================================

Keys are taken from everything after ``/cache/`` and percent-decoded, so keys
containing slashes work either as ``/cache/a/b`` or as ``/cache/a%2Fb``.

This is a demo server, not a production deployment: there is no authentication,
no TLS, no rate limiting, and a thread per connection. Bind it to localhost.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

from .cache import MISS, LRUCache

CACHE_PREFIX = "/cache/"
STATS_PATH = "/stats"

#: Requests with a larger body are refused rather than buffered.
MAX_BODY_BYTES = 1 << 20

#: How much of an over-long body will be read and discarded so that a 413 can be
#: delivered on a connection that stays usable. Beyond this the connection is
#: dropped instead: draining without limit would let a client make the server
#: read as much as it cares to send.
MAX_DRAIN_BYTES = 8 * MAX_BODY_BYTES

_DRAIN_CHUNK = 1 << 16

_CACHE_METHODS = "GET, PUT, DELETE, OPTIONS"
_STATS_METHODS = "GET, OPTIONS"


class CacheServer(ThreadingHTTPServer):
    """A threaded HTTP server bound to one cache instance."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        cache: LRUCache[str, Any],
        *,
        log_requests: bool = True,
    ) -> None:
        """Bind a server for ``cache``.

        Args:
            address: ``(host, port)`` to bind. Port 0 picks a free one, which is
                what the tests use to avoid collisions.
            cache: The cache to expose. Shared across request threads, which is
                safe because the cache locks internally.
            log_requests: Whether to log one line per request to stderr.
        """
        # Set before binding so a handler can never observe a server without one.
        self.cache = cache
        self.log_requests = log_requests
        super().__init__(address, CacheRequestHandler)

    @property
    def url(self) -> str:
        """The base URL this server is reachable on, e.g. ``http://127.0.0.1:8080``."""
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


class CacheRequestHandler(BaseHTTPRequestHandler):
    """Translates HTTP requests into cache operations."""

    server_version = "lrucache"
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------

    @property
    def cache(self) -> LRUCache[str, Any]:
        """The cache owned by the server handling this request."""
        return self.server.cache  # type: ignore[attr-defined,no-any-return]

    def log_message(self, format: str, *args: Any) -> None:
        """Log one line per request, unless the server was asked to stay quiet."""
        if getattr(self.server, "log_requests", True):
            super().log_message(format, *args)

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        """Send ``payload`` as a JSON body with an accurate Content-Length."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_no_content(self) -> None:
        """Send a bodiless 204."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def _send_problem(self, status: HTTPStatus, detail: str) -> None:
        """Send an error as JSON, so clients only ever have to parse one format."""
        self._send_json(status, {"error": status.phrase, "detail": detail})

    def _route_key(self) -> str | None:
        """Return the cache key this request addresses, or None if it is not a key route."""
        path = urlsplit(self.path).path
        if not path.startswith(CACHE_PREFIX):
            return None
        key = unquote(path[len(CACHE_PREFIX) :])
        return key or None

    def _reject_method(self) -> None:
        """Answer a request whose path is known but whose method is not allowed."""
        path = urlsplit(self.path).path
        if path == STATS_PATH:
            allowed = _STATS_METHODS
        elif path.startswith(CACHE_PREFIX):
            allowed = _CACHE_METHODS
        else:
            self._send_problem(HTTPStatus.NOT_FOUND, f"no route for {path}")
            return

        body = json.dumps(
            {
                "error": HTTPStatus.METHOD_NOT_ALLOWED.phrase,
                "detail": f"{self.command} is not allowed on {path}",
            }
        ).encode("utf-8")
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Allow", allowed)
        self.end_headers()
        self.wfile.write(body)

    def _drain_body(self, length: int) -> bool:
        """Read and discard a request body, in chunks, without buffering it.

        Returns True if the whole declared body was consumed, meaning the
        connection is still in a usable state. False if it was too long to be
        worth draining, or the client stopped sending early.
        """
        if length > MAX_DRAIN_BYTES:
            return False

        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(_DRAIN_CHUNK, remaining))
            if not chunk:
                return False
            remaining -= len(chunk)
        return True

    def _read_json_object(self) -> dict[str, Any] | None:
        """Read and parse a JSON object body.

        Returns None after having already sent an error response, so callers can
        simply bail out. The declared body is always consumed in full when it is
        read at all, otherwise leftover bytes would be parsed as the start of the
        next request on a keep-alive connection.
        """
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_problem(HTTPStatus.LENGTH_REQUIRED, "a Content-Length header is required")
            return None

        try:
            length = int(raw_length)
        except ValueError:
            self._send_problem(HTTPStatus.BAD_REQUEST, f"malformed Content-Length {raw_length!r}")
            return None

        if length < 0:
            self._send_problem(HTTPStatus.BAD_REQUEST, "Content-Length cannot be negative")
            return None
        if length > MAX_BODY_BYTES:
            # The body has to be consumed before the 413 can be written, even
            # though it is being thrown away: a client that is still uploading
            # will never read a response on a connection the server has already
            # closed underneath it. Past MAX_DRAIN_BYTES that stops being worth
            # it and the connection is dropped instead.
            if not self._drain_body(length):
                self.close_connection = True
            self._send_problem(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"body of {length} bytes exceeds the {MAX_BODY_BYTES} byte limit",
            )
            return None

        body = self.rfile.read(length)
        if not body:
            self._send_problem(HTTPStatus.BAD_REQUEST, "a JSON body is required")
            return None

        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_problem(HTTPStatus.BAD_REQUEST, f"body is not valid JSON: {exc}")
            return None

        if not isinstance(payload, dict):
            self._send_problem(
                HTTPStatus.BAD_REQUEST,
                f"body must be a JSON object, got {type(payload).__name__}",
            )
            return None

        return payload

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:
        """Serve ``/stats`` and cache reads."""
        path = urlsplit(self.path).path
        if path == STATS_PATH:
            self._send_json(HTTPStatus.OK, self.cache.stats().as_dict())
            return

        key = self._route_key()
        if key is None:
            self._send_problem(HTTPStatus.NOT_FOUND, f"no route for {path}")
            return

        value = self.cache.get(key, MISS)
        if value is MISS:
            self._send_problem(HTTPStatus.NOT_FOUND, f"key {key!r} is not cached")
            return

        self._send_json(HTTPStatus.OK, {"key": key, "value": value})

    def do_PUT(self) -> None:
        """Store a value, optionally with a TTL.

        Always answers 204 rather than distinguishing a create from a replace:
        checking first would mean a second lock acquisition and a race with other
        writers, for information a cache client has no real use for.
        """
        key = self._route_key()
        if key is None:
            self._reject_method()
            return

        payload = self._read_json_object()
        if payload is None:
            return

        if "value" not in payload:
            self._send_problem(HTTPStatus.BAD_REQUEST, "body must contain a 'value' field")
            return

        ttl = payload.get("ttl")
        if ttl is not None and (not isinstance(ttl, int | float) or isinstance(ttl, bool)):
            self._send_problem(HTTPStatus.BAD_REQUEST, f"ttl must be a number, got {ttl!r}")
            return

        try:
            self.cache.put(key, payload["value"], ttl)
        except ValueError as exc:
            self._send_problem(HTTPStatus.BAD_REQUEST, str(exc))
            return

        self._send_no_content()

    def do_DELETE(self) -> None:
        """Remove a key, reporting whether there was anything fresh to remove."""
        key = self._route_key()
        if key is None:
            self._reject_method()
            return

        if self.cache.delete(key):
            self._send_no_content()
        else:
            self._send_problem(HTTPStatus.NOT_FOUND, f"key {key!r} is not cached")

    def do_OPTIONS(self) -> None:
        """Advertise the methods a route supports."""
        path = urlsplit(self.path).path
        if path == STATS_PATH:
            allowed = _STATS_METHODS
        elif path.startswith(CACHE_PREFIX):
            allowed = _CACHE_METHODS
        else:
            self._send_problem(HTTPStatus.NOT_FOUND, f"no route for {path}")
            return

        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", allowed)
        self.end_headers()

    # HEAD is deliberately absent rather than wired to _reject_method: a response
    # to HEAD must carry no body, and an error document here would be read by the
    # client as the start of the next response.
    do_POST = _reject_method
    do_PATCH = _reject_method


def create_server(
    cache: LRUCache[str, Any],
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    log_requests: bool = True,
) -> CacheServer:
    """Bind and return a :class:`CacheServer` without starting it.

    Separating binding from serving is what lets the tests start a server on an
    ephemeral port, read back the port that was actually assigned, and shut it
    down cleanly.
    """
    return CacheServer((host, port), cache, log_requests=log_requests)
