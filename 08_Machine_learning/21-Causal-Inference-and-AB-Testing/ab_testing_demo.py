from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from random import Random
from statistics import mean


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


@dataclass
class ExperimentResult:
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    z_score: float
    p_value: float


def analyze_ab_test(control_successes: int, control_total: int, treatment_successes: int, treatment_total: int) -> ExperimentResult:
    control_rate = control_successes / control_total
    treatment_rate = treatment_successes / treatment_total
    pooled_rate = (control_successes + treatment_successes) / (control_total + treatment_total)
    standard_error = sqrt(pooled_rate * (1 - pooled_rate) * ((1 / control_total) + (1 / treatment_total)))
    z_score = (treatment_rate - control_rate) / standard_error
    p_value = 2 * (1 - normal_cdf(abs(z_score)))
    return ExperimentResult(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        absolute_lift=treatment_rate - control_rate,
        z_score=z_score,
        p_value=p_value,
    )


def stratified_ate(rows: list[dict[str, int]], treatment_key: str, outcome_key: str, strata_key: str) -> float:
    strata: dict[int, list[dict[str, int]]] = {}
    for row in rows:
        strata.setdefault(row[strata_key], []).append(row)

    weighted_effects: list[float] = []
    weights: list[int] = []
    for stratum_rows in strata.values():
        treated = [row[outcome_key] for row in stratum_rows if row[treatment_key] == 1]
        control = [row[outcome_key] for row in stratum_rows if row[treatment_key] == 0]
        if not treated or not control:
            continue
        weighted_effects.append(mean(treated) - mean(control))
        weights.append(len(stratum_rows))
    return sum(effect * weight for effect, weight in zip(weighted_effects, weights)) / sum(weights)


def simulate_observational_data(seed: int = 7) -> list[dict[str, int]]:
    rng = Random(seed)
    rows: list[dict[str, int]] = []
    for _ in range(500):
        intent_bucket = rng.choice([0, 1, 2])
        treatment_probability = 0.20 + 0.25 * intent_bucket
        treated = 1 if rng.random() < treatment_probability else 0
        conversion_probability = 0.08 + 0.05 * intent_bucket + 0.03 * treated
        converted = 1 if rng.random() < conversion_probability else 0
        rows.append({"intent_bucket": intent_bucket, "treated": treated, "converted": converted})
    return rows


def main() -> None:
    result = analyze_ab_test(control_successes=210, control_total=2000, treatment_successes=255, treatment_total=2000)
    print("A/B test summary")
    print(f"- control_rate   = {result.control_rate:.4f}")
    print(f"- treatment_rate = {result.treatment_rate:.4f}")
    print(f"- absolute_lift  = {result.absolute_lift:.4f}")
    print(f"- z_score        = {result.z_score:.3f}")
    print(f"- p_value        = {result.p_value:.4f}")

    observational_rows = simulate_observational_data()
    ate = stratified_ate(observational_rows, "treated", "converted", "intent_bucket")
    print("\nObservational estimate")
    print(f"- stratified_ate = {ate:.4f}")


if __name__ == "__main__":
    main()
