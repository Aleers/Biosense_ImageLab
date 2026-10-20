"""Optional time-based smoothing; never fills missing observations."""
from dataclasses import dataclass
import numpy as np
from landmark_quality import LandmarkValidator

@dataclass
class TemporalLandmarkFilter:
    time_constant_seconds: float = 0.04
    visibility_threshold: float = 0.45
    max_gap_seconds: float = 0.1

    def __post_init__(self):
        if self.time_constant_seconds <= 0 or self.max_gap_seconds <= 0:
            raise ValueError("Filter durations must be positive")
        self.previous = None
        self.timestamp = None

    def update(self, points, timestamp):
        if not np.isfinite(timestamp):
            raise ValueError("Invalid timestamp")
        if points is None:
            self.previous = None
            self.timestamp = timestamp
            return None
        result = points.copy()
        dt = None if self.timestamp is None else timestamp - self.timestamp
        if self.previous is not None and dt is not None and 0 < dt <= self.max_gap_seconds:
            validator = LandmarkValidator(self.visibility_threshold)
            alpha = 1 - np.exp(-dt / self.time_constant_seconds)
            for index in range(len(points)):
                if validator.valid(points, (index,)) and validator.valid(self.previous, (index,)):
                    result[index, :2] = alpha * points[index, :2] + (1 - alpha) * self.previous[index, :2]
        self.previous = result.copy()
        self.timestamp = timestamp
        return result
