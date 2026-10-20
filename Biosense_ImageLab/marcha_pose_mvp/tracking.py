"""Correction tracking independent of Tkinter and shared mutable analysis data."""
from dataclasses import dataclass, field, asdict
from enum import Enum
import cv2
import numpy as np

class TrackingState(str, Enum):
    MANUAL = "manual"
    TRACKING = "tracking"
    SUSPECT = "suspect"
    LOST = "lost"
    REACQUIRED = "reacquired"

@dataclass(frozen=True)
class TrackingOptions:
    search_radius: int = 40
    max_speed_px_per_second: float = 900.0
    minimum_similarity: float = 0.72
    ambiguity_margin: float = 0.08
    reacquire_frames: int = 2
    use_optical_flow: bool = False
    smoothing_seconds: float = 0.0
    max_prediction_seconds: float = 0.15

@dataclass
class TrackingBatch:
    updates: list = field(default_factory=list)
    records: list = field(default_factory=list)
    cancelled: bool = False

class LandmarkTracker:
    def __init__(self, options=None):
        self.options = options or TrackingOptions()
        if (self.options.reacquire_frames < 1 or self.options.search_radius < 1
                or not 0 <= self.options.minimum_similarity <= 1
                or self.options.smoothing_seconds < 0 or self.options.max_speed_px_per_second <= 0):
            raise ValueError("Invalid tracking options")

    @staticmethod
    def patch(gray, position, radius=10):
        x, y = np.asarray(position, dtype=float)
        if not np.all(np.isfinite((x, y))) or not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]):
            return None
        return cv2.getRectSubPix(gray, (2 * radius + 1, 2 * radius + 1), (float(x), float(y)))

    @staticmethod
    def similarity(template, patch):
        if template is None or patch is None or np.std(template) < 2:
            return 0.0
        error = float(cv2.matchTemplate(patch, template, cv2.TM_SQDIFF_NORMED)[0, 0])
        return max(0.0, 1.0 - error) if np.isfinite(error) else 0.0

    def select_marker(self, frame, prediction, template, initial_area=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        radius = self.options.search_radius
        cx, cy = map(int, prediction)
        x0, x1 = max(0, cx-radius), min(frame.shape[1], cx+radius+1)
        y0, y1 = max(0, cy-radius), min(frame.shape[0], cy+radius+1)
        if x0 >= x1 or y0 >= y1:
            return None, "empty_search_region", None
        mask = cv2.inRange(hsv[y0:y1, x0:x1], np.array([0, 0, 205]), np.array([180, 75, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for n in range(1, count):
            area, width, height = stats[n, cv2.CC_STAT_AREA], stats[n, cv2.CC_STAT_WIDTH], stats[n, cv2.CC_STAT_HEIGHT]
            if not (8 <= area <= 1200 and 0.55 <= width / max(height, 1) <= 1.8):
                continue
            if initial_area is not None and not 0.5 <= area / initial_area <= 2.0:
                continue
            point = centers[n] + (x0, y0)
            distance = float(np.linalg.norm(point - prediction))
            if distance > radius:
                continue
            similarity = self.similarity(template, self.patch(gray, point))
            if similarity < self.options.minimum_similarity:
                continue
            score = similarity - 0.2 * distance / radius
            candidates.append((score, point, float(area)))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            return None, "no_matching_marker", None
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < self.options.ambiguity_margin:
            return None, "competing_targets", None
        return candidates[0][1], "white_marker", candidates[0][2]

    def optical_prediction(self, previous, current, point):
        source = np.asarray(point, np.float32).reshape(1, 1, 2)
        forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, source, None)
        if forward is None or not status[0, 0]:
            return None
        backward, status, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None)
        if backward is None or not status[0, 0] or np.linalg.norm(backward-source) > 1.5:
            return None
        return forward[0, 0].astype(float)

    def run(self, path, landmarks, start, landmark, position, fps, cancelled=None):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid FPS")
        cap = cv2.VideoCapture(str(path))
        batch = TrackingBatch()
        try:
            if not cap.isOpened():
                raise ValueError("Nao foi possivel abrir o video para rastreamento.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ok, initial = cap.read()
            if not ok:
                raise ValueError(f"Falha de leitura no quadro {start}.")
            previous = np.asarray(position, dtype=float).copy()
            gray_previous = cv2.cvtColor(initial, cv2.COLOR_BGR2GRAY)
            template = self.patch(gray_previous, previous)
            if template is None:
                raise ValueError("Correcao fora da imagem.")
            initial_value = landmarks[start][landmark].copy()
            initial_value[:2], initial_value[2] = previous, 1.0
            batch.updates.append((start, initial_value))
            batch.records.append(dict(frame=start, landmark=landmark, state=TrackingState.MANUAL.value, source="manual", reason="user_correction"))
            _, _, initial_area = self.select_marker(initial, previous, template)
            factory = getattr(cv2, 'TrackerCSRT_create', None) or getattr(getattr(cv2, 'legacy', None), 'TrackerCSRT_create', None)
            tracker = None
            if factory:
                tracker = factory()
                size = min(36, initial.shape[0], initial.shape[1])
                bx = max(0, min(initial.shape[1]-size, int(previous[0]-size/2)))
                by = max(0, min(initial.shape[0]-size, int(previous[1]-size/2)))
                tracker.init(initial, (bx, by, size, size))
            batch.records[0]['csrt_available'] = tracker is not None
            batch.records[0]['options'] = asdict(self.options)
            velocity = np.zeros(2)
            misses = 0
            reacquire_count = 0
            pending = None
            max_jump = max(8.0, self.options.max_speed_px_per_second / fps)
            for index in range(start+1, len(landmarks)):
                if cancelled is not None and cancelled.is_set():
                    batch.cancelled = True
                    return batch
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Falha de leitura no quadro {index}; correcao nao aplicada.")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                horizon = min(misses+1, max(1, int(fps*self.options.max_prediction_seconds)))
                prediction = previous + velocity*horizon
                tracked, box = tracker.update(frame) if tracker else (False, None)
                candidate, reason, _ = self.select_marker(frame, prediction, template, initial_area)
                ambiguous = reason == "competing_targets"
                source = "white_marker"
                if candidate is None and tracked and not ambiguous:
                    bx, by, width, height = box
                    point = np.array([bx+width/2, by+height/2])
                    if self.similarity(template, self.patch(gray, point)) >= self.options.minimum_similarity:
                        candidate, source, reason = point, "csrt", "appearance_verified"
                if candidate is None and self.options.use_optical_flow and not ambiguous and misses == 0:
                    point = self.optical_prediction(gray_previous, gray, previous)
                    if point is not None and self.similarity(template, self.patch(gray, point)) >= self.options.minimum_similarity:
                        candidate, source, reason = point, "optical_flow", "forward_backward_verified"
                if candidate is not None and np.linalg.norm(candidate-prediction) > max_jump:
                    candidate, reason = None, "implausible_jump"
                state = TrackingState.TRACKING
                if candidate is None:
                    misses += 1
                    reacquire_count, pending = 0, None
                    state = TrackingState.SUSPECT if ambiguous else TrackingState.LOST
                elif misses:
                    if pending is not None and np.linalg.norm(candidate-pending) <= max_jump:
                        reacquire_count += 1
                    else:
                        reacquire_count = 1
                    pending = candidate.copy()
                    if reacquire_count < self.options.reacquire_frames:
                        candidate = None
                        state = TrackingState.SUSPECT
                        reason = "awaiting_reacquisition"
                        misses += 1
                    else:
                        state = TrackingState.REACQUIRED
                if candidate is not None:
                    if self.options.smoothing_seconds > 0 and state == TrackingState.TRACKING:
                        alpha = 1-np.exp(-1/(fps*self.options.smoothing_seconds))
                        candidate = alpha*candidate+(1-alpha)*previous
                    velocity = (candidate-previous)/max(1, misses+1)
                    previous = candidate.copy()
                    misses = 0
                future = landmarks[index]
                if future is not None:
                    value = future[landmark].copy()
                    if candidate is None:
                        value[2] = 0.0
                    else:
                        value[:2] = candidate
                        value[2] = 0.8  # Acceptance threshold, not calibrated confidence.
                    batch.updates.append((index, value))
                batch.records.append(dict(frame=index, landmark=landmark, state=state.value,
                    source=source if candidate is not None else "unavailable", reason=reason,
                    position=None if candidate is None else candidate.tolist()))
                gray_previous = gray
            return batch
        finally:
            cap.release()
