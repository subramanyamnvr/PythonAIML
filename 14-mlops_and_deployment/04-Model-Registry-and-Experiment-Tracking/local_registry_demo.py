from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RunRecord:
    run_id: str
    model_name: str
    metric_name: str
    metric_value: float
    stage: str


class LocalModelRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load(self) -> list[RunRecord]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [RunRecord(**row) for row in payload]

    def save(self, records: list[RunRecord]) -> None:
        self.path.write_text(json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8")

    def register(self, record: RunRecord) -> None:
        records = self.load()
        records.append(record)
        self.save(records)

    def best_run(self, metric_name: str) -> RunRecord:
        records = [record for record in self.load() if record.metric_name == metric_name]
        return max(records, key=lambda item: item.metric_value)


def main() -> None:
    registry = LocalModelRegistry(Path(__file__).with_name("registry.json"))
    registry.save([])
    registry.register(RunRecord("run-001", "baseline_xgb", "f1", 0.78, "staging"))
    registry.register(RunRecord("run-002", "feature_set_b", "f1", 0.81, "staging"))
    registry.register(RunRecord("run-003", "calibrated_model", "f1", 0.80, "candidate"))

    best = registry.best_run("f1")
    print("Best registered run")
    print(f"- run_id     = {best.run_id}")
    print(f"- model_name = {best.model_name}")
    print(f"- metric     = {best.metric_name}:{best.metric_value}")
    print(f"- stage      = {best.stage}")


if __name__ == "__main__":
    main()
