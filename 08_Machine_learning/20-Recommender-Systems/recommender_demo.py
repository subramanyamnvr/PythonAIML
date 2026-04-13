from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Dict


Ratings = Dict[str, Dict[str, float]]


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    common_items = set(left) & set(right)
    if not common_items:
        return 0.0
    numerator = sum(left[item] * right[item] for item in common_items)
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def user_based_recommendations(ratings: Ratings, target_user: str, top_k: int = 3) -> list[tuple[str, float]]:
    target_profile = ratings[target_user]
    candidate_scores: defaultdict[str, float] = defaultdict(float)
    similarity_weights: defaultdict[str, float] = defaultdict(float)

    for other_user, other_profile in ratings.items():
        if other_user == target_user:
            continue
        similarity = cosine_similarity(target_profile, other_profile)
        if similarity <= 0:
            continue
        for item, score in other_profile.items():
            if item in target_profile:
                continue
            candidate_scores[item] += similarity * score
            similarity_weights[item] += similarity

    ranked: list[tuple[str, float]] = []
    for item, weighted_score in candidate_scores.items():
        ranked.append((item, weighted_score / similarity_weights[item]))
    return sorted(ranked, key=lambda row: row[1], reverse=True)[:top_k]


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    shortlist = recommended[:k]
    hits = sum(1 for item in shortlist if item in relevant)
    return hits / k


def build_sample_ratings() -> Ratings:
    return {
        "alice": {"matrix": 5, "arrival": 4, "blade_runner": 4},
        "bob": {"matrix": 5, "arrival": 5, "interstellar": 4, "dune": 4},
        "carol": {"blade_runner": 5, "interstellar": 4, "dune": 5},
        "dan": {"matrix": 4, "blade_runner": 4, "dune": 5, "gravity": 3},
        "eve": {"arrival": 5, "gravity": 4, "interstellar": 5},
    }


def main() -> None:
    ratings = build_sample_ratings()
    recommendations = user_based_recommendations(ratings, target_user="alice", top_k=3)
    print("Recommendations for alice")
    for item, score in recommendations:
        print(f"- {item:14s} predicted_rating={score:.2f}")

    recommended_items = [item for item, _ in recommendations]
    held_out_relevant = {"dune", "interstellar"}
    print(
        "\nEvaluation",
        f"precision@3={precision_at_k(recommended_items, held_out_relevant, 3):.2f}",
    )


if __name__ == "__main__":
    main()
