"""Command line entry point: ``python -m lrucache``.

Configuration comes from three places, and the first one that supplies a value
wins: a command line flag, then an environment variable, then the built-in
default. That ordering is what lets a container set ``LRU_CAPACITY`` for the
normal case while an operator still overrides it for one run.

===================== ==================== =============
Flag                  Environment variable Default
===================== ==================== =============
``--capacity``        ``LRU_CAPACITY``     128
``--host``            ``LRU_HOST``         127.0.0.1
``--port``            ``LRU_PORT``         8080
``--default-ttl``     ``LRU_DEFAULT_TTL``  none
===================== ==================== =============
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from . import __version__
from .cache import LRUCache
from .server import create_server

DEFAULT_CAPACITY = 128
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

ENV_CAPACITY = "LRU_CAPACITY"
ENV_HOST = "LRU_HOST"
ENV_PORT = "LRU_PORT"
ENV_DEFAULT_TTL = "LRU_DEFAULT_TTL"

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a flag or environment variable holds an unusable value."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """A fully resolved server configuration."""

    capacity: int = DEFAULT_CAPACITY
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    default_ttl: float | None = None


def positive_int(raw: str) -> int:
    """Parse a string as an integer of at least 1."""
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{raw!r} is not an integer") from None
    if value < 1:
        raise ValueError(f"must be at least 1, got {value}")
    return value


def port_number(raw: str) -> int:
    """Parse a string as a TCP port, allowing 0 to mean "any free port"."""
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{raw!r} is not an integer") from None
    if not 0 <= value <= 65535:
        raise ValueError(f"must be between 0 and 65535, got {value}")
    return value


def positive_float(raw: str) -> float:
    """Parse a string as a number of seconds greater than zero."""
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{raw!r} is not a number") from None
    if value <= 0:
        raise ValueError(f"must be greater than 0, got {value}")
    return value


def _from_env(
    env: Mapping[str, str], name: str, converter: Callable[[str], T], fallback: T
) -> T:
    """Read ``name`` from ``env``, falling back when it is unset or blank.

    A variable that is set but unusable is an error rather than something to
    quietly ignore: silently falling back to a default capacity would be a
    confusing way to learn about a typo in a deployment config.
    """
    raw = env.get(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        return converter(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"environment variable {name}={raw!r} is invalid: {exc}") from exc


def build_parser(defaults: ServerConfig) -> argparse.ArgumentParser:
    """Build the argument parser, using ``defaults`` for unspecified flags."""
    parser = argparse.ArgumentParser(
        prog="python -m lrucache",
        description="Serve an in-memory LRU cache over HTTP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--capacity",
        type=positive_int,
        default=defaults.capacity,
        help=f"maximum number of entries (env: {ENV_CAPACITY})",
    )
    parser.add_argument(
        "--host",
        default=defaults.host,
        help=f"interface to bind (env: {ENV_HOST})",
    )
    parser.add_argument(
        "--port",
        type=port_number,
        default=defaults.port,
        help=f"port to bind, 0 for any free port (env: {ENV_PORT})",
    )
    parser.add_argument(
        "--default-ttl",
        type=positive_float,
        default=defaults.default_ttl,
        metavar="SECONDS",
        help=f"expire entries after this long unless a PUT says otherwise "
        f"(env: {ENV_DEFAULT_TTL})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not log a line per request",
    )
    parser.add_argument("--version", action="version", version=f"lrucache {__version__}")
    return parser


def resolve_config(
    argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None
) -> tuple[ServerConfig, bool]:
    """Resolve the configuration from flags, then environment, then defaults.

    Args:
        argv: Command line arguments, excluding the program name.
        env: Environment mapping. Defaults to the real environment.

    Returns:
        The resolved config, and whether request logging was suppressed.

    Raises:
        ConfigError: If an environment variable holds an unusable value.
        SystemExit: If a command line flag is unusable, via argparse.
    """
    env = os.environ if env is None else env

    from_env = ServerConfig(
        capacity=_from_env(env, ENV_CAPACITY, positive_int, DEFAULT_CAPACITY),
        host=_from_env(env, ENV_HOST, str, DEFAULT_HOST),
        port=_from_env(env, ENV_PORT, port_number, DEFAULT_PORT),
        default_ttl=_from_env(env, ENV_DEFAULT_TTL, positive_float, None),
    )

    args = build_parser(from_env).parse_args(argv)
    config = ServerConfig(
        capacity=args.capacity,
        host=args.host,
        port=args.port,
        default_ttl=args.default_ttl,
    )
    return config, bool(args.quiet)


def main(argv: Sequence[str] | None = None) -> int:
    """Serve the cache until interrupted. Returns a process exit code."""
    try:
        config, quiet = resolve_config(argv)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cache: LRUCache[str, Any] = LRUCache(config.capacity, default_ttl=config.default_ttl)
    server = create_server(cache, config.host, config.port, log_requests=not quiet)

    ttl = "none" if config.default_ttl is None else f"{config.default_ttl}s"
    print(
        f"lrucache {__version__} serving on {server.url} "
        f"(capacity={config.capacity}, default_ttl={ttl})",
        file=sys.stderr,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
    finally:
        server.server_close()

    stats = cache.stats()
    print(
        f"final stats: {stats.hits} hits, {stats.misses} misses, "
        f"{stats.evictions} evictions, hit rate {stats.hit_rate:.1%}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
