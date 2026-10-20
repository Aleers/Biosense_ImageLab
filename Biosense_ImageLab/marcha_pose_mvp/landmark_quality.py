"""Shared quality rules for 2D landmarks and measurements."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class QualityAssessment:
    valid: bool
    reasons: tuple[str, ...] = ()

@dataclass(frozen=True)
class LandmarkValidator:
    visibility_threshold: float = 0.45

    def assess(self, points, indices):
        if points is None:
            return QualityAssessment(False, ("person_missing",))
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] < 3:
            return QualityAssessment(False, ("invalid_shape",))
        if any(i < 0 or i >= len(points) for i in indices):
            return QualityAssessment(False, ("invalid_index",))
        selected = points[list(indices), :3]
        if not np.all(np.isfinite(selected)):
            return QualityAssessment(False, ("nonfinite_landmark",))
        if np.any(selected[:, 2] < self.visibility_threshold):
            return QualityAssessment(False, ("low_visibility",))
        if np.any(selected[:, 2] > 1):
            return QualityAssessment(False, ("invalid_visibility",))
        return QualityAssessment(True)

    def valid(self, points, indices):
        return self.assess(points, indices).valid

    @staticmethod
    def scale(points, centimeters):
        if points is None or centimeters is None:
            return None
        array = np.asarray(points, dtype=float)
        if array.shape != (2, 2) or not np.all(np.isfinite(array)):
            return None
        if not np.isfinite(centimeters) or centimeters <= 0:
            return None
        distance = float(np.linalg.norm(array[0] - array[1]))
        return distance / centimeters if distance >= 1 else None

    def distance(self, points, indices, px_per_cm=None, horizontal=False):
        if len(indices) != 2 or not self.valid(points, indices):
            return None
        if px_per_cm is not None and (not np.isfinite(px_per_cm) or px_per_cm <= 0):
            return None
        delta = points[indices[0], :2] - points[indices[1], :2]
        value = float(abs(delta[0]) if horizontal else np.linalg.norm(delta))
        return value / px_per_cm if px_per_cm is not None else value
