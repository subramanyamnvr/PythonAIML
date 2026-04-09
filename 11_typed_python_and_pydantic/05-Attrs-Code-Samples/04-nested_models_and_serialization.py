"""Lesson 4: nested attrs models.

This file shows how attrs models can be composed and then converted
back into plain dictionaries with asdict.
"""

from __future__ import annotations

from attrs import asdict, define, field


@define
class Address:
    city: str
    country: str


@define
class Customer:
    customer_id: int
    name: str
    address: Address
    tags: list[str] = field(factory=list)


if __name__ == "__main__":
    customer = Customer(1, "Ava", Address("Bengaluru", "India"), ["vip"])
    print(asdict(customer))
