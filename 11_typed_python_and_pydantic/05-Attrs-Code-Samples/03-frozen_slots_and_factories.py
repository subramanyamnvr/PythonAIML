"""Lesson 3: immutable attrs models.

This file shows how attrs can model configuration-like data
that should not change after creation.
"""

from __future__ import annotations

from attrs import field, frozen


@frozen
class CacheSettings:
    prefix: str
    timeout_seconds: int = 300
    tags: tuple[str, ...] = field(factory=tuple)


if __name__ == "__main__":
    settings = CacheSettings("users", tags=("fast", "demo"))
    print(settings)
