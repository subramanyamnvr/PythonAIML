from __future__ import annotations

from dataclasses import asdict, dataclass
from json import dumps
from statistics import mean


@dataclass
class DatasetCard:
    name: str
    task: str
    num_samples: int
    label_distribution: dict[str, int]


@dataclass
class ExperimentConfig:
    model_name: str
    image_size: int
    confidence_threshold: float
    iou_threshold: float


@dataclass
class EvaluationSummary:
    precision: float
    recall: float
    f1: float
    notes: list[str]


def compute_metrics(tp: int, fp: int, fn: int) -> EvaluationSummary:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
    notes = [
        "check class imbalance before comparing variants",
        "inspect false positives by lighting and camera angle",
        "review low-confidence misses separately from total misses",
    ]
    return EvaluationSummary(precision=precision, recall=recall, f1=f1, notes=notes)


def build_project_report(dataset: DatasetCard, config: ExperimentConfig, evaluation: EvaluationSummary) -> str:
    payload = {
        "dataset": asdict(dataset),
        "experiment": asdict(config),
        "evaluation": asdict(evaluation),
    }
    return dumps(payload, indent=2)


def main() -> None:
    dataset = DatasetCard(
        name="warehouse_safety_demo",
        task="helmet_detection",
        num_samples=1250,
        label_distribution={"helmet": 710, "no_helmet": 540},
    )
    config = ExperimentConfig(
        model_name="yolo_like_baseline",
        image_size=640,
        confidence_threshold=0.35,
        iou_threshold=0.5,
    )
    evaluation = compute_metrics(tp=148, fp=19, fn=24)
    print(build_project_report(dataset, config, evaluation))
    print("\nHigh-level score")
    print(f"- mean_metric = {mean([evaluation.precision, evaluation.recall, evaluation.f1]):.3f}")


if __name__ == "__main__":
    main()
