"""Lesson 2: aliases and literals.

This file shows how to:
- give domain meaning to primitive types with TypeAlias
- restrict a value to a known safe set with Literal
"""

from __future__ import annotations

from typing import Literal, TypeAlias

# These aliases improve readability even though they resolve to simple types.
Environment = Literal["local", "staging", "prod"]
UserId: TypeAlias = int


def build_profile_url(environment: Environment, user_id: UserId) -> str:
    host_map = {
        "local": "http://localhost:8000",
        "staging": "https://staging.example.com",
        "prod": "https://example.com",
    }
    return f"{host_map[environment]}/users/{user_id}"


if __name__ == "__main__":
    url = build_profile_url("staging", 42)
    print(url)
