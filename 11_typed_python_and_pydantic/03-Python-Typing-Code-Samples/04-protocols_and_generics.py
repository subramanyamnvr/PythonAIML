"""Lesson 4: protocols and generics.

This file demonstrates structural typing:
- objects do not need a shared base class
- they only need to satisfy the required interface
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, TypeVar


class HasId(Protocol):
    # Any object with an integer id matches this protocol.
    id: int


T = TypeVar("T", bound=HasId)


def collect_ids(items: Iterable[T]) -> list[int]:
    return [item.id for item in items]


@dataclass
class User:
    id: int
    name: str


@dataclass
class Order:
    id: int
    total: float


if __name__ == "__main__":
    ids = collect_ids([User(1, "Ava"), Order(2, 19.99)])
    print(ids)
