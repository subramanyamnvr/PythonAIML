"""Lesson 5: overloads and type guards.

This file shows how to improve type narrowing for helper functions
that accept more than one input shape.
"""

from __future__ import annotations

from typing import Iterable, TypeGuard, overload


@overload
def ensure_list(value: None) -> list[str]:
    ...


@overload
def ensure_list(value: str) -> list[str]:
    ...


@overload
def ensure_list(value: Iterable[str]) -> list[str]:
    ...


def ensure_list(value: None | str | Iterable[str]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def is_str_list(value: list[object]) -> TypeGuard[list[str]]:
    # A successful runtime check lets type checkers narrow the list type.
    return all(isinstance(item, str) for item in value)


if __name__ == "__main__":
    result = ensure_list(("python", "pydantic"))
    maybe_strings: list[object] = ["a", "b", "c"]

    print(result)
    if is_str_list(maybe_strings):
        print(",".join(maybe_strings))
