"""Lesson 5: nested dataclasses and serialization.

This file demonstrates how multiple dataclasses can compose into
a larger object graph and then be converted into plain dictionaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Address:
    city: str
    country: str


@dataclass
class Customer:
    customer_id: int
    name: str
    address: Address
    tags: list[str] = field(default_factory=list)


if __name__ == "__main__":
    customer = Customer(1, "Ava", Address("Bengaluru", "India"), ["vip"])
    # asdict recursively converts nested dataclasses into built-in types.
    print(asdict(customer))
