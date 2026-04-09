"""Lesson 1: the simplest dataclass.

This file shows how dataclasses remove constructor and repr boilerplate
for plain structured objects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UserProfile:
    user_id: int
    name: str
    email: str


if __name__ == "__main__":
    profile = UserProfile(1, "Ava", "ava@example.com")
    print(profile)
