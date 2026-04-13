from __future__ import annotations

from dataclasses import dataclass
from math import dist


BoundingBox = tuple[int, int, int, int]


@dataclass
class Track:
    track_id: int
    centroid: tuple[float, float]
    age: int = 0


def centroid(box: BoundingBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


class CentroidTracker:
    def __init__(self, max_distance: float = 35.0, max_age: int = 2) -> None:
        self.max_distance = max_distance
        self.max_age = max_age
        self.next_track_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[BoundingBox]) -> dict[int, tuple[float, float]]:
        detection_centroids = [centroid(box) for box in detections]
        unmatched_tracks = set(self.tracks)
        unmatched_detections = set(range(len(detection_centroids)))

        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for detection_index, detection_center in enumerate(detection_centroids):
                pairs.append((dist(track.centroid, detection_center), track_id, detection_index))

        for distance_value, track_id, detection_index in sorted(pairs):
            if distance_value > self.max_distance:
                continue
            if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            self.tracks[track_id].centroid = detection_centroids[detection_index]
            self.tracks[track_id].age = 0
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_index)

        for track_id in list(unmatched_tracks):
            self.tracks[track_id].age += 1
            if self.tracks[track_id].age > self.max_age:
                del self.tracks[track_id]

        for detection_index in unmatched_detections:
            self.tracks[self.next_track_id] = Track(self.next_track_id, detection_centroids[detection_index])
            self.next_track_id += 1

        return {track_id: track.centroid for track_id, track in sorted(self.tracks.items())}


def demo_frames() -> list[list[BoundingBox]]:
    return [
        [(10, 10, 30, 30), (90, 10, 110, 30)],
        [(15, 14, 35, 34), (86, 12, 106, 32)],
        [(20, 18, 40, 38)],
        [(26, 22, 46, 42), (82, 16, 102, 36)],
    ]


def main() -> None:
    tracker = CentroidTracker()
    for frame_index, detections in enumerate(demo_frames(), start=1):
        assignments = tracker.update(detections)
        print(f"Frame {frame_index}")
        for track_id, track_centroid in assignments.items():
            print(f"- track={track_id} centroid={track_centroid}")


if __name__ == "__main__":
    main()
