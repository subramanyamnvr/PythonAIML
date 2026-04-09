# Python Typing Code Samples

This folder is a guided mini-path for the standard-library `typing` tools that sit underneath modern Python data modeling.

If Pydantic feels like the "runtime validation" layer, `typing` is the language used to describe the shape of that data first.

## Why This Folder Exists

- Pydantic models rely heavily on type annotations.
- Libraries like `dataclasses`, `attrs`, and `msgspec` also benefit from strong type hints.
- Understanding `typing` makes later Pydantic concepts like unions, literals, generics, and annotated types much easier.

## What You Should Learn Here

- how Python type hints describe function inputs and outputs
- how aliases and `Literal` narrow allowed values
- how `TypedDict` models dictionary-shaped data
- how `Protocol` and generics describe reusable interfaces
- how `overload` and `TypeGuard` improve static reasoning

## Recommended Order

1. `01-basic_type_hints.py`
2. `02-type_aliases_and_literals.py`
3. `03-typed_dicts_and_unions.py`
4. `04-protocols_and_generics.py`
5. `05-overload_and_type_guard.py`

## File Guide

### `01-basic_type_hints.py`

- Introduces simple parameter and return annotations.
- Shows why `Iterable[str]` is broader than `list[str]`.
- Good first step for understanding how type hints document intent.

### `02-type_aliases_and_literals.py`

- Shows how to create meaningful named types with `TypeAlias`.
- Uses `Literal` to restrict values to a small safe set.
- This pattern is common in config, environment, and API code.

### `03-typed_dicts_and_unions.py`

- Demonstrates dictionary-shaped schemas with `TypedDict`.
- Uses `NotRequired` and union syntax to model optional input.
- This is a useful bridge toward request payload validation.

### `04-protocols_and_generics.py`

- Introduces structural typing with `Protocol`.
- Shows a generic function that works for multiple object types as long as they share an `id`.
- Helps explain why many typed APIs stay flexible without inheritance.

### `05-overload_and_type_guard.py`

- Demonstrates how one function can expose different typed call patterns.
- Shows how `TypeGuard` narrows types after a runtime check.
- Useful when building helper utilities around parsed or partially trusted data.

## How This Connects To Pydantic

- Pydantic reads these annotations to know what shape your model expects.
- The more comfortable you are with `typing`, the easier advanced Pydantic features feel.
- `Literal`, unions, typed containers, and annotated types appear regularly in real Pydantic models.

## How To Run

Run any sample directly:

- `python 01-basic_type_hints.py`
- `python 02-type_aliases_and_literals.py`
- `python 03-typed_dicts_and_unions.py`
- `python 04-protocols_and_generics.py`
- `python 05-overload_and_type_guard.py`
