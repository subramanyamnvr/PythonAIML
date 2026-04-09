"""Lesson 1: a minimal attrs model.

This file shows how attrs creates clean, compact classes with very little code.
"""

from __future__ import annotations

from attrs import define


@define
class ServiceConfig:
    host: str
    port: int = 8000
    debug: bool = False


if __name__ == "__main__":
    config = ServiceConfig("localhost", debug=True)
    print(config)
