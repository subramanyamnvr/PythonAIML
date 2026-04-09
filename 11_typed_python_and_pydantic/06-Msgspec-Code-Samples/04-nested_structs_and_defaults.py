"""Lesson 4: nested structs and built-in conversion.

This file demonstrates nested msgspec models and a final conversion
back to regular Python objects for downstream use.
"""

from __future__ import annotations

import msgspec


class Address(msgspec.Struct):
    city: str
    country: str


class Customer(msgspec.Struct):
    customer_id: int
    name: str
    address: Address
    tags: tuple[str, ...] = ()


if __name__ == "__main__":
    customer = Customer(
        1,
        "Ava",
        Address("Bengaluru", "India"),
        ("vip", "beta"),
    )
    # to_builtins converts the typed model into plain dicts, tuples, and scalars.
    print(msgspec.to_builtins(customer))
