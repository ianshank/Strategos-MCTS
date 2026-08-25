"""JSONL cassette format for record/replay of LLM responses.

Each line is a JSON object:

.. code-block:: json

    {"request_hash": "<hex>", "request": {...}, "response": {...}}

The hash is computed from a *canonical* serialisation of the request so that
keys with cosmetic differences (ordering, whitespace) don't fragment matching.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CassetteEntry:
    """A single recorded request/response pair."""

    request_hash: str
    request: dict[str, Any]
    response: dict[str, Any]


def hash_request(request: dict[str, Any]) -> str:
    """Compute a stable SHA-256 hash of a canonicalised request payload."""
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Cassette:
    """Append-only cassette stored as a JSONL file.

    The class is *deliberately small* — it only knows how to append, look
    up by hash, and iterate. Concurrency is the caller's responsibility.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._index: dict[str, CassetteEntry] | None = None

    @property
    def path(self) -> Path:
        """The on-disk JSONL location."""
        return self._path

    def _load_index(self) -> dict[str, CassetteEntry]:
        """Lazy-load and cache the file contents."""
        if self._index is None:
            self._index = {}
            if self._path.exists():
                with self._path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        entry = CassetteEntry(
                            request_hash=record["request_hash"],
                            request=record["request"],
                            response=record["response"],
                        )
                        self._index[entry.request_hash] = entry
        return self._index

    def lookup(self, request_hash: str) -> CassetteEntry | None:
        """Return the recorded entry for ``request_hash`` or ``None``."""
        return self._load_index().get(request_hash)

    def append(self, request: dict[str, Any], response: dict[str, Any]) -> CassetteEntry:
        """Persist a new entry. Returns the appended entry."""
        request_hash = hash_request(request)
        entry = CassetteEntry(request_hash=request_hash, request=request, response=response)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "request_hash": entry.request_hash,
                        "request": entry.request,
                        "response": entry.response,
                    },
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
        if self._index is not None:
            self._index[request_hash] = entry
        return entry

    def __iter__(self) -> Iterator[CassetteEntry]:
        return iter(self._load_index().values())

    def __len__(self) -> int:
        return len(self._load_index())


__all__ = ["Cassette", "CassetteEntry", "hash_request"]
