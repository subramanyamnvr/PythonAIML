from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RunManifest:
    run_name: str
    seed: int
    data_hash: str
    config_hash: str
    python_version: str


def stable_hash(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


def deterministic_split(items: list[int], seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    values = items[:]
    rng.shuffle(values)
    split = int(len(values) * 0.8)
    return values[:split], values[split:]


def build_manifest(run_name: str, seed: int, data: list[int], config: dict[str, object]) -> RunManifest:
    return RunManifest(
        run_name=run_name,
        seed=seed,
        data_hash=stable_hash(data),
        config_hash=stable_hash(config),
        python_version=platform.python_version(),
    )


def save_manifest(path: Path, manifest: RunManifest) -> None:
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


def load_manifest(path: Path) -> RunManifest:
    return RunManifest(**json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    data = list(range(1, 21))
    config = {"model": "baseline", "learning_rate": 0.01, "epochs": 5}
    train_a, valid_a = deterministic_split(data, seed=42)
    train_b, valid_b = deterministic_split(data, seed=42)
    manifest = build_manifest("baseline_run", seed=42, data=data, config=config)
    manifest_path = Path(__file__).with_name("run_manifest.json")
    save_manifest(manifest_path, manifest)

    print("Deterministic split check")
    print(f"- identical_train_split={train_a == train_b}")
    print(f"- identical_valid_split={valid_a == valid_b}")

    print("\nSaved manifest")
    print(load_manifest(manifest_path))


if __name__ == "__main__":
    main()
