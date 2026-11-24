"""One request at a time, and replay of answered requests, for ``POST /wm``."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

Key = tuple[str, int]  # (robot_episode_session_id, req_id)


def request_digest(request: dict[str, Any]) -> str:
    """Hash of everything that decides a request's output, as canonical JSON."""

    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Admission:
    kind: str  # "run", "cached" (replay ``cached``), "busy" or "conflict" (the same key with other inputs)
    cached: dict[str, Any] | None = None


class WmRequestGate:
    """Admits one request at a time and replays the last answered ones by key."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: tuple[Key | None, str] | None = None  # (key, digest) of the request being generated
        self._done: OrderedDict[Key, tuple[str, dict[str, Any]]] = OrderedDict()

    @staticmethod
    def key_of(payload: dict[str, Any]) -> Key | None:
        """The replay key of a request; ``req_id`` -1 is never replayed."""

        if payload["req_id"] == -1:
            return None
        return str(payload["robot_episode_session_id"]), payload["req_id"]

    def try_begin(self, key: Key | None, digest: str) -> Admission:
        with self._lock:
            if key is not None and key in self._done:
                done_digest, response = self._done[key]
                if done_digest != digest:
                    return Admission("conflict")
                self._done.move_to_end(key)
                return Admission("cached", dict(response))
            if self._running is not None:
                running_key, running_digest = self._running
                same_request_changed = key is not None and running_key == key and running_digest != digest
                return Admission("conflict" if same_request_changed else "busy")
            self._running = key, digest
            return Admission("run")

    def commit(self, key: Key | None, digest: str, response: dict[str, Any]) -> None:
        with self._lock:
            self._running = None
            if key is not None:
                self._done[key] = digest, dict(response)
                while len(self._done) > 4:  # answered requests kept for replay
                    self._done.popitem(last=False)

    def abort(self) -> None:
        with self._lock:
            self._running = None

    def reset(self) -> None:
        """Forget the answered requests; a running one finishes."""

        with self._lock:
            self._done.clear()

    @property
    def busy(self) -> bool:
        return self._running is not None
