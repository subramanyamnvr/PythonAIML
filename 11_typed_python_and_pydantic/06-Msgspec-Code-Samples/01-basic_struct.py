"""Lesson 1: a minimal msgspec struct.

This file introduces the shape of a msgspec model before any encoding or conversion.
"""

from __future__ import annotations

import msgspec


class User(msgspec.Struct):
    id: int
    name: str
    active: bool = True


if __name__ == "__main__":
    user = User(1, "Ava")
    print(user)
