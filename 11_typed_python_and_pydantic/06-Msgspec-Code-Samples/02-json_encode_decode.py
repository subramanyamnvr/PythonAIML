"""Lesson 2: JSON round-tripping with msgspec.

This file shows how a typed struct can be serialized to bytes
and then restored as a typed object.
"""

from __future__ import annotations

import msgspec


class Job(msgspec.Struct):
    name: str
    retries: int = 0


if __name__ == "__main__":
    payload = msgspec.json.encode(Job("sync-search-index", 2))
    # Passing type=Job restores a strongly typed object instead of a raw dict.
    restored = msgspec.json.decode(payload, type=Job)

    print(payload)
    print(restored)
