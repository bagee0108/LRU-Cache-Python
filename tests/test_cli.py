"""Tests for configuration resolution: flags, then environment, then defaults."""

from __future__ import annotations

import contextlib
import io
import unittest

from lrucache.__main__ import (
    DEFAULT_CAPACITY,
    DEFAULT_HOST,
    DEFAULT_PORT,
    ConfigError,
    port_number,
    positive_float,
    positive_int,
    resolve_config,
)


def config(argv: list[str] | None = None, **env: str):
    """Resolve a config from the given flags and a fully controlled environment."""
    resolved, _quiet = resolve_config(argv or [], env)
    return resolved


class ConverterTests(unittest.TestCase):
    def test_positive_int_accepts_a_positive_integer(self) -> None:
        self.assertEqual(positive_int("42"), 42)

    def test_positive_int_rejects_zero_and_negatives(self) -> None:
        for raw in ("0", "-1"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    positive_int(raw)

    def test_positive_int_rejects_non_numbers(self) -> None:
        for raw in ("abc", "1.5", ""):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    positive_int(raw)

    def test_port_number_accepts_the_full_range_including_zero(self) -> None:
        self.assertEqual(port_number("0"), 0)
        self.assertEqual(port_number("65535"), 65535)

    def test_port_number_rejects_out_of_range_values(self) -> None:
        for raw in ("-1", "65536"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    port_number(raw)

    def test_positive_float_accepts_fractions(self) -> None:
        self.assertEqual(positive_float("0.5"), 0.5)

    def test_positive_float_rejects_zero_and_negatives(self) -> None:
        for raw in ("0", "-2.5"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    positive_float(raw)


class DefaultsTests(unittest.TestCase):
    def test_nothing_supplied_yields_the_documented_defaults(self) -> None:
        resolved = config()

        self.assertEqual(resolved.capacity, DEFAULT_CAPACITY)
        self.assertEqual(resolved.host, DEFAULT_HOST)
        self.assertEqual(resolved.port, DEFAULT_PORT)
        self.assertIsNone(resolved.default_ttl)

    def test_logging_is_on_unless_quiet_is_passed(self) -> None:
        _, quiet = resolve_config([], {})
        self.assertFalse(quiet)

        _, quiet = resolve_config(["--quiet"], {})
        self.assertTrue(quiet)


class EnvironmentTests(unittest.TestCase):
    def test_environment_supplies_every_setting(self) -> None:
        resolved = config(
            LRU_CAPACITY="512",
            LRU_HOST="0.0.0.0",
            LRU_PORT="9000",
            LRU_DEFAULT_TTL="30",
        )

        self.assertEqual(resolved.capacity, 512)
        self.assertEqual(resolved.host, "0.0.0.0")
        self.assertEqual(resolved.port, 9000)
        self.assertEqual(resolved.default_ttl, 30.0)

    def test_a_blank_environment_variable_falls_back_to_the_default(self) -> None:
        self.assertEqual(config(LRU_CAPACITY="   ").capacity, DEFAULT_CAPACITY)

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        self.assertEqual(config(LRU_CAPACITY=" 64 ").capacity, 64)

    def test_an_invalid_environment_variable_is_an_error_not_a_silent_default(self) -> None:
        """A typo in a deployment config must be loud, not absorbed."""
        with self.assertRaises(ConfigError) as caught:
            config(LRU_CAPACITY="lots")

        self.assertIn("LRU_CAPACITY", str(caught.exception))

    def test_each_invalid_environment_variable_is_reported(self) -> None:
        for name, bad in (
            ("LRU_CAPACITY", "0"),
            ("LRU_PORT", "99999"),
            ("LRU_DEFAULT_TTL", "-5"),
        ):
            with self.subTest(variable=name):
                with self.assertRaises(ConfigError):
                    config(**{name: bad})


class PrecedenceTests(unittest.TestCase):
    def test_a_flag_overrides_the_environment(self) -> None:
        resolved = config(["--capacity", "10"], LRU_CAPACITY="512")

        self.assertEqual(resolved.capacity, 10)

    def test_unspecified_flags_still_come_from_the_environment(self) -> None:
        resolved = config(["--capacity", "10"], LRU_CAPACITY="512", LRU_PORT="9000")

        self.assertEqual(resolved.capacity, 10)
        self.assertEqual(resolved.port, 9000)

    def test_every_setting_can_be_overridden_by_a_flag(self) -> None:
        resolved = config(
            ["--capacity", "1", "--host", "localhost", "--port", "1234", "--default-ttl", "2.5"],
            LRU_CAPACITY="512",
            LRU_HOST="0.0.0.0",
            LRU_PORT="9000",
            LRU_DEFAULT_TTL="30",
        )

        self.assertEqual(resolved.capacity, 1)
        self.assertEqual(resolved.host, "localhost")
        self.assertEqual(resolved.port, 1234)
        self.assertEqual(resolved.default_ttl, 2.5)


class FlagValidationTests(unittest.TestCase):
    """argparse reports a bad flag and exits 2, which is the conventional behaviour."""

    def assert_exits_with_usage(self, argv: list[str]) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            resolve_config(argv, {})

        self.assertEqual(caught.exception.code, 2)
        self.assertIn("usage:", stderr.getvalue())

    def test_zero_capacity_is_rejected(self) -> None:
        self.assert_exits_with_usage(["--capacity", "0"])

    def test_non_numeric_capacity_is_rejected(self) -> None:
        self.assert_exits_with_usage(["--capacity", "lots"])

    def test_out_of_range_port_is_rejected(self) -> None:
        self.assert_exits_with_usage(["--port", "70000"])

    def test_non_positive_ttl_is_rejected(self) -> None:
        self.assert_exits_with_usage(["--default-ttl", "0"])

    def test_unknown_flag_is_rejected(self) -> None:
        self.assert_exits_with_usage(["--nope"])

    def test_help_exits_successfully(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as caught:
            resolve_config(["--help"], {})

        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--capacity", stdout.getvalue())


class MainErrorHandlingTests(unittest.TestCase):
    def test_a_bad_environment_variable_exits_2_without_a_traceback(self) -> None:
        import os

        from lrucache.__main__ import main

        stderr = io.StringIO()
        original = os.environ.get("LRU_CAPACITY")
        os.environ["LRU_CAPACITY"] = "not-a-number"
        try:
            with contextlib.redirect_stderr(stderr):
                exit_code = main([])
        finally:
            if original is None:
                del os.environ["LRU_CAPACITY"]
            else:
                os.environ["LRU_CAPACITY"] = original

        self.assertEqual(exit_code, 2)
        self.assertIn("LRU_CAPACITY", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
