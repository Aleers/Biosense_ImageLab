"""Pure 2D joint geometry, independent of rendering and MediaPipe imports."""
from dataclasses import dataclass
import math
import numpy as np
from landmark_quality import LandmarkValidator

JOINT_DEFINITIONS = {
    "hip": ("SHOULDER", "HIP", "KNEE"),
    "knee": ("HIP", "KNEE", "ANKLE"),
    "ankle": ("KNEE", "ANKLE", "FOOT_INDEX"),
}
# Stable MediaPipe Pose landmark indices.
LANDMARK_IDS = {
    "left": {"SHOULDER": 11, "HIP": 23, "KNEE": 25, "ANKLE": 27, "FOOT_INDEX": 31},
    "right": {"SHOULDER": 12, "HIP": 24, "KNEE": 26, "ANKLE": 28, "FOOT_INDEX": 32},
}

def angle_2d(a, b, c):
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    if any(p.shape != (2,) or not np.all(np.isfinite(p)) for p in (a, b, c)):
        return float("nan")
    ba, bc = a - b, c - b
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-8 or not np.isfinite(den):
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(ba, bc) / den, -1, 1))))

@dataclass(frozen=True)
class JointAngleCalculator:
    visibility_threshold: float = 0.45

    def calculate(self, points, sides=("left", "right")):
        metrics = {}
        if points is None:
            return metrics
        for side in sides:
            for joint, names in JOINT_DEFINITIONS.items():
                indices = [LANDMARK_IDS[side][name] for name in names]
                if not LandmarkValidator(self.visibility_threshold).valid(points, indices):
                    continue
                selected = points[indices, :3]
                value = angle_2d(*selected[:, :2])
                if math.isfinite(value):
                    metrics[f"{side}_{joint}_deg"] = value
        return metrics

def calculate_joint_angles(points, sides=("left", "right"), visibility_threshold=0.45):
    return JointAngleCalculator(visibility_threshold).calculate(points, sides)
