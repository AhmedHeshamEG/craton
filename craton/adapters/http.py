"""A polite HTTP client: one request at a time per host, a pause between them, retries.

Data providers are public services run by small teams. Craton asks for each file once, caches
it (``craton.store.raw``), identifies itself in the User-Agent, waits between requests and backs
off when a server struggles.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Protocol

from craton import __version__

USER_AGENT = f"craton-earth/{__version__} (+https://github.com/AhmedHeshamEG/craton)"


class HttpError(RuntimeError):
    """The server answered with an error status."""

    def __init__(self, url: str, status: int, body: str) -> None:
        self.url, self.status, self.body = url, status, body
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")


class Fetcher(Protocol):
    """Anything that turns a URL into bytes; tests pass a dictionary-backed one."""

    def get(self, url: str) -> bytes: ...


class Http:
    """The real fetcher."""

    def __init__(
        self,
        min_interval_s: float = 1.0,
        retries: int = 3,
        timeout_s: float = 180.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.retries = retries
        self.timeout_s = timeout_s
        self._sleep = sleep
        self._last: dict[str, float] = {}

    def _wait_turn(self, host: str) -> None:
        last = self._last.get(host)
        if last is not None:
            gap = time.monotonic() - last
            if gap < self.min_interval_s:
                self._sleep(self.min_interval_s - gap)
        self._last[host] = time.monotonic()

    def get(self, url: str) -> bytes:
        host = urllib.parse.urlsplit(url).netloc
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        for attempt in range(self.retries + 1):
            self._wait_turn(host)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    data: bytes = response.read()
                    return data
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                if exc.code < 500 or attempt == self.retries:
                    raise HttpError(url, exc.code, body) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries:
                    raise
            self._sleep(5.0 * 2**attempt)
        raise AssertionError("unreachable")
