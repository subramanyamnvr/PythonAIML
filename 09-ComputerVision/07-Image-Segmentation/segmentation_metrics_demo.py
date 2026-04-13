from __future__ import annotations

from typing import Iterable


Mask = list[list[int]]


def threshold_image(image: list[list[int]], threshold: int) -> Mask:
    return [[1 if pixel >= threshold else 0 for pixel in row] for row in image]


def flatten(mask: Mask) -> list[int]:
    return [value for row in mask for value in row]


def intersection_over_union(prediction: Mask, truth: Mask) -> float:
    pred_flat = flatten(prediction)
    truth_flat = flatten(truth)
    intersection = sum(1 for pred, target in zip(pred_flat, truth_flat) if pred == 1 and target == 1)
    union = sum(1 for pred, target in zip(pred_flat, truth_flat) if pred == 1 or target == 1)
    return intersection / union if union else 1.0


def dice_score(prediction: Mask, truth: Mask) -> float:
    pred_flat = flatten(prediction)
    truth_flat = flatten(truth)
    intersection = sum(1 for pred, target in zip(pred_flat, truth_flat) if pred == 1 and target == 1)
    positives = sum(pred_flat) + sum(truth_flat)
    return (2 * intersection) / positives if positives else 1.0


def pixel_accuracy(prediction: Mask, truth: Mask) -> float:
    pred_flat = flatten(prediction)
    truth_flat = flatten(truth)
    correct = sum(1 for pred, target in zip(pred_flat, truth_flat) if pred == target)
    return correct / len(pred_flat)


def print_mask(title: str, mask: Mask) -> None:
    print(title)
    for row in mask:
        print(" ".join(str(value) for value in row))


def main() -> None:
    image = [
        [10, 15, 20, 18, 12],
        [14, 190, 205, 182, 18],
        [12, 200, 220, 188, 16],
        [10, 175, 196, 181, 15],
        [8, 12, 18, 11, 9],
    ]
    ground_truth = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    prediction = threshold_image(image, threshold=180)

    print_mask("Predicted mask", prediction)
    print()
    print(f"IoU           = {intersection_over_union(prediction, ground_truth):.3f}")
    print(f"Dice score    = {dice_score(prediction, ground_truth):.3f}")
    print(f"Pixel accuracy= {pixel_accuracy(prediction, ground_truth):.3f}")


if __name__ == "__main__":
    main()
