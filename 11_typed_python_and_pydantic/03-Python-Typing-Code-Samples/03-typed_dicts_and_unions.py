"""Lesson 3: TypedDict and union types.

This file models dictionary-shaped payload data before moving into full validation tools.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class UserPayload(TypedDict):
    id: int
    name: str
    email: str
    # The key may exist, but it is not required in every payload.
    is_admin: NotRequired[bool]


def describe_user(payload: UserPayload | None) -> str:
    if payload is None:
        return "No payload supplied"

    role = "admin" if payload.get("is_admin", False) else "member"
    return f"{payload['name']} <{payload['email']}> is a {role}"


if __name__ == "__main__":
    sample: UserPayload = {
        "id": 1,
        "name": "Ava",
        "email": "ava@example.com",
    }
    print(describe_user(sample))
