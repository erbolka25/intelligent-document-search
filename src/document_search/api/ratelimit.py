"""In-memory sliding-window rate limiter (single process, resets on restart)."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    """Allow at most ``limit`` hits per ``window_s`` seconds for each key."""

    def __init__(
        self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> int:
        """Record a hit. Returns 0 if allowed, else seconds until the next allowed hit."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(1, math.ceil(hits[0] + self.window_s - now))
            hits.append(now)
            if len(self._hits) > 10_000:  # forget idle keys so memory stays bounded
                self._prune(now)
            return 0

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)

    def _prune(self, now: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= now - self.window_s]:
            del self._hits[key]
