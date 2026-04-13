# Typed Python + Pydantic Path

This folder sits before GenAI in the main repo order and now covers typed Python foundations, related data-model libraries, and the existing Pydantic learning path.

## What These Libraries Are

- `typing`: Python's standard type-hinting system for function signatures, generics, unions, protocols, and static analysis.
- `dataclasses`: Python's standard lightweight data-container module for structured classes with automatic `__init__`, `__repr__`, and equality support.
- `attrs`: A flexible third-party library for structured classes with validators, converters, immutability, and ergonomic model definitions.
- `msgspec`: A fast structured-data library for typed models plus efficient JSON and MessagePack encoding or decoding.
- `pydantic`: A data parsing and validation library that uses type hints to validate external input at runtime and convert it into reliable Python objects.

## Why They Are Used

- `typing` is used to make code easier to understand, navigate, and statically check in editors and tools like `mypy` or `pyright`.
- `dataclasses` is used when you want simple structured objects for internal app state without heavy runtime validation.
- `attrs` is used when you want dataclass-like models with stronger ergonomics and richer validation or conversion features.
- `msgspec` is used when performance matters and you need fast typed serialization or deserialization.
- `pydantic` is used when data comes from APIs, forms, environment variables, config files, or LLM outputs and must be validated at runtime.

## Quick Comparison

| Tool | Main Role | Runtime Validation | Best For |
| --- | --- | --- | --- |
| `typing` | Type hints and static contracts | No | Readability, IDE support, static checking |
| `dataclasses` | Lightweight structured classes | Minimal | Internal models and clean Python objects |
| `attrs` | Flexible structured classes | Yes, via validators | Richer object modeling without full schema tooling |
| `msgspec` | Fast typed serialization | Yes | High-performance APIs and data pipelines |
| `pydantic` | Parsing and validation | Yes | External data, settings, schemas, API payloads |

## When To Use Which

- Use `typing` when you want better code quality and clearer contracts without changing runtime behavior.
- Use `dataclasses` when the data is already trusted and you mainly want clean object structure.
- Use `attrs` when you want a more powerful dataclass-like model with validators and converters.
- Use `msgspec` when speed and efficient serialization are a priority.
- Use `pydantic` when correctness of incoming data matters more than lightweight object creation.

## Current Order

- `01-Pydantic-Project`
- `02-Basics`
- `03-Python-Typing-Code-Samples`
- `04-Dataclasses-Code-Samples`
- `05-Attrs-Code-Samples`
- `06-Msgspec-Code-Samples`
- `07-Model-Configuration`
- `08-Field-Aliasing-Serialization-and-Deserialization`
- `09-Specialized-Pydantic-Types`
- `10-Additional-Field-Features`
- `11-Annotated-Types`
- `12-Custom-Validators`
- `13-Properties-and-Computed-Fields`
- `14-Custom-Serializers-Using-Annotated-Types`
- `15-Complex-Models`
- `16-Applications`
- `17-Pydantic-Essentials-Meta`

## How To Use It

- Keep projects and notes in numbered folders so the section stays easy to sort.
- Start `01-Pydantic-Project/onepydantic` if you want one integrated runnable example before the notebook-heavy path.
- Use `03` through `06` for companion code samples covering stdlib typing, dataclasses, and nearby libraries.
- Use `02` and `07` through `16` as the main Pydantic course path.
- Use `17-Pydantic-Essentials-Meta` for the imported course README, dependency files, and license.
- Open `module_map.md` for the audited file order inside each folder.
