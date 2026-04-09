# Dataclasses Code Samples

This folder is a guided introduction to Python's built-in `dataclasses` module.

`dataclasses` is often the first stop when you want structured objects in Python without writing repetitive boilerplate.

## Why This Folder Exists

- It shows the difference between simple structured objects and full runtime validation.
- It helps explain what Pydantic adds on top of plain Python classes.
- It gives you a lighter-weight option for trusted internal data.

## What You Should Learn Here

- how `@dataclass` generates constructor and repr methods
- how `default_factory` avoids shared mutable defaults
- how `__post_init__` adds custom validation or derived values
- how `frozen`, `slots`, and inheritance shape object behavior
- how nested dataclasses serialize with `asdict`

## Recommended Order

1. `01-basic_dataclass.py`
2. `02-default_factory_and_order.py`
3. `03-post_init_and_validation.py`
4. `04-frozen_slots_and_inheritance.py`
5. `05-nested_models_and_asdict.py`

## File Guide

### `01-basic_dataclass.py`

- The simplest dataclass example.
- Good for seeing how much boilerplate disappears immediately.

### `02-default_factory_and_order.py`

- Shows list defaults done safely with `default_factory`.
- Adds ordering support so instances can be sorted.

### `03-post_init_and_validation.py`

- Demonstrates derived fields and manual validation.
- This is the closest plain-dataclass equivalent to simple model validation.

### `04-frozen_slots_and_inheritance.py`

- Introduces immutability with `frozen=True`.
- Shows memory and attribute restrictions with `slots=True`.
- Includes inheritance to demonstrate how dataclass hierarchies behave.

### `05-nested_models_and_asdict.py`

- Models nested data structures.
- Shows how to turn dataclass objects into plain dictionaries.

## Dataclasses vs Pydantic

- Use `dataclasses` when your data is already trusted and you mainly want clean structure.
- Use Pydantic when the input may be messy, external, or needs runtime coercion and validation.
- Dataclasses are lighter; Pydantic is safer for real-world inputs.

## How To Run

- `python 01-basic_dataclass.py`
- `python 02-default_factory_and_order.py`
- `python 03-post_init_and_validation.py`
- `python 04-frozen_slots_and_inheritance.py`
- `python 05-nested_models_and_asdict.py`
