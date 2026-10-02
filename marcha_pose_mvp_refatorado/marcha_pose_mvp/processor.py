from __future__ import annotations
import csv, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import cv2, mediapipe as mp, numpy as np
from i18n import tr

def text_width(text, scale, thickness=2):
    """Calcula a largura usando a mesma fonte ASCII desenhada pelo OpenCV."""
    return cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]

def dark_panel(frame, p1, p2):
    x0, y0 = (max(0, p1[0]), max(0, p1[1]))
    x1, y1 = (min(frame.shape[1], p2[0]), min(frame.shape[0], p2[1]))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    shade = np.empty_like(roi)
    shade[:] = (15, 15, 15)
    cv2.addWeighted(shade, 0.8, roi, 0.2, 0, roi)

@dataclass(frozen=True)
class ProcessingOptions:
    """Configuracoes imutaveis escolhidas antes do processamento."""
    view: str = 'Lateral'
    side: str = 'Automatico'
    visibility_threshold: float = 0.45
    language: str = 'pt'

@dataclass
class Analysis:
    """Estado compartilhado entre deteccao, revisao e exportacao.

    Guardar os dados em um unico objeto evita variaveis globais e facilita a
    justificativa do fluxo do programa durante a apresentacao do TCC.
    """
    input_path: Path
    fps: float
    width: int
    height: int
    landmarks: list[np.ndarray | None] = field(default_factory=list)
    calibration_points: tuple[tuple[float, float], tuple[float, float]] | None = None
    calibration_cm: float | None = None
    predominant_side: str = 'left'
    manual_segments: list[dict] = field(default_factory=list)
    tracked_segments: list[dict] = field(default_factory=list)

    @property
    def px_per_cm(self):
        if not self.calibration_points or not self.calibration_cm:
            return None
        a, b = map(np.asarray, self.calibration_points)
        return float(np.linalg.norm(a - b) / self.calibration_cm)

def angle(a, b, c):
    ba, bc = (np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b))
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    return float('nan') if den < 1e-08 else math.degrees(math.acos(float(np.clip(np.dot(ba, bc) / den, -1, 1))))

class PoseVideoAnalyzer:
    """Le o video e converte cada quadro em landmarks do MediaPipe.

    Esta classe possui uma unica responsabilidade: executar a deteccao. As
    correcoes manuais e a exportacao permanecem em etapas separadas.
    """

    def __init__(self, language='pt', progress=None):
        self.language = language
        self.progress = progress

    def analyze(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError('Nao foi possivel abrir o video.')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        data = Analysis(path, fps, width, height)
        pose = mp.solutions.pose
        with pose.Pose(model_complexity=2, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
            frame_number = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                found = result.pose_landmarks
                points = None
                if found:
                    points = np.array([[p.x * width, p.y * height, p.visibility] for p in found.landmark], dtype=np.float32)
                data.landmarks.append(points)
                frame_number += 1
                self._report_progress(frame_number, total)
        cap.release()
        if not data.landmarks:
            raise ValueError('O video nao contem quadros validos.')
        stabilize_lower_limbs(data)
        return data

    def _report_progress(self, frame_number, total):
        if self.progress and frame_number % 5 == 0:
            message = f"{tr(self.language, 'detecting')}: {frame_number} {tr(self.language, 'of')} {total}..."
            self.progress(min(99, frame_number * 100 / total), message)

def analyze_video(path: Path, progress: Callable | None=None, language='pt'):
    """Fachada mantida para compatibilidade com a interface existente."""
    return PoseVideoAnalyzer(language, progress).analyze(path)

def stabilize_lower_limbs(data):
    """Mantem a identidade esquerda/direita e fixa o lado predominante do video."""
    pose = mp.solutions.pose.PoseLandmark
    names = ('HIP', 'KNEE', 'ANKLE', 'HEEL', 'FOOT_INDEX')
    pairs = [(int(getattr(pose, f'LEFT_{n}')), int(getattr(pose, f'RIGHT_{n}'))) for n in names]
    scores = {'left': [], 'right': []}
    previous = None
    for points in data.landmarks:
        if points is None:
            continue
        scores['left'].append(np.mean([points[a, 2] for a, _ in pairs]))
        scores['right'].append(np.mean([points[b, 2] for _, b in pairs]))
        if previous is not None:
            normal = sum((np.linalg.norm(points[a, :2] - previous[a, :2]) + np.linalg.norm(points[b, :2] - previous[b, :2]) for a, b in pairs))
            swapped = sum((np.linalg.norm(points[b, :2] - previous[a, :2]) + np.linalg.norm(points[a, :2] - previous[b, :2]) for a, b in pairs))
            if swapped < normal * 0.72:
                for a, b in pairs:
                    points[[a, b]] = points[[b, a]]
            for a, b in pairs:
                for idx in (a, b):
                    if points[idx, 2] < 0.42 and previous[idx, 2] >= 0.42:
                        points[idx, :2] = previous[idx, :2]
                        points[idx, 2] = 0.42
        previous = points.copy()
    data.predominant_side = max(scores, key=lambda s: np.median(scores[s]) if scores[s] else 0)

def gait_phases(data):
    """Heuristica 2D: proximidade do solo + baixa velocidade do pe."""
    pose = mp.solutions.pose.PoseLandmark
    result = {}
    for side in ('left', 'right'):
        heel = int(getattr(pose, f'{side.upper()}_HEEL'))
        toe = int(getattr(pose, f'{side.upper()}_FOOT_INDEX'))
        centers = []
        lowest = []
        for p in data.landmarks:
            if p is None:
                centers.append(None)
                lowest.append(np.nan)
            else:
                centers.append((p[heel, :2] + p[toe, :2]) / 2)
                lowest.append(max(p[heel, 1], p[toe, 1]))
        valid = np.array([v for v in lowest if np.isfinite(v)])
        ground = np.percentile(valid, 90) if len(valid) else data.height
        speed = np.zeros(len(centers))
        for i in range(1, len(centers)):
            if centers[i] is not None and centers[i - 1] is not None:
                speed[i] = np.linalg.norm(centers[i] - centers[i - 1])
        moving = speed[speed > 0]
        speed_limit = np.percentile(moving, 55) if len(moving) else 8
        raw = np.array([np.isfinite(y) and y > ground - data.height * 0.045 and (speed[i] <= max(5, speed_limit)) for i, y in enumerate(lowest)], dtype=float)
        smooth = np.convolve(raw, np.ones(7) / 7, mode='same') >= 0.5
        events = [''] * len(smooth)
        for i in range(1, len(smooth)):
            if smooth[i] and (not smooth[i - 1]):
                events[i] = 'CONTATO INICIAL'
            elif not smooth[i] and smooth[i - 1]:
                events[i] = 'RETIRADA DO PE'
        result[side] = {'stance': smooth, 'event': events, 'centers': centers}
    return result

def draw_angle_table(frame, current, running, lang='pt'):
    keys = [k for k in current if k.endswith('_deg') and any((tag in k for tag in ('_hip_', '_knee_', '_ankle_', '_q_estimated_')))]
    if not keys:
        return
    labels = {'left': 'E' if lang != 'en' else 'L', 'right': 'D' if lang != 'en' else 'R', 'hip': tr(lang, 'hip'), 'knee': tr(lang, 'knee'), 'ankle': tr(lang, 'ankle'), 'q': tr(lang, 'q')}
    rows = []
    for key in sorted(keys):
        parts = key.split('_')
        name = f'{labels.get(parts[0], parts[0])} {labels.get(parts[1], parts[1])}'
        v = current[key]
        mn, mx = running[key]
        rows.append((name, v, mn, mx))
    name_w = max([text_width(r[0], 0.61, 2) for r in rows] + [text_width(tr(lang, 'measure_col'), 0.61, 2)])
    x_name = 20
    x_current = x_name + name_w + 34
    x_min = x_current + 94
    x_max = x_min + 88
    x2 = min(frame.shape[1] - 8, x_max + 78)
    bottom = 134 + 32 * len(rows)
    dark_panel(frame, (8, 58), (x2, bottom))
    cv2.putText(frame, tr(lang, 'angles'), (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
    for text, x in ((tr(lang, 'measure_col'), x_name), (tr(lang, 'current'), x_current), (tr(lang, 'min'), x_min), (tr(lang, 'max'), x_max)):
        cv2.putText(frame, text, (x, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (220, 225, 235), 2, cv2.LINE_AA)
    for row, (name, v, mn, mx) in enumerate(rows):
        y = 148 + 32 * row
        cv2.putText(frame, name, (x_name, y), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (235, 242, 255), 2, cv2.LINE_AA)
        for value, x in ((v, x_current), (mn, x_min), (mx, x_max)):
            cv2.putText(frame, f'{value:.1f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (235, 242, 255), 2, cv2.LINE_AA)

def stepdown_metrics(points, px_per_cm=None):
    if points is None:
        return {}
    pose = mp.solutions.pose.PoseLandmark
    out = {}
    lh, rh = (int(pose.LEFT_HIP), int(pose.RIGHT_HIP))
    ls, rs = (int(pose.LEFT_SHOULDER), int(pose.RIGHT_SHOULDER))
    pelvis = points[rh, :2] - points[lh, :2]
    out['pelvic_tilt_deg'] = math.degrees(math.atan2(pelvis[1], pelvis[0]))
    shoulder_mid = (points[ls, :2] + points[rs, :2]) / 2
    hip_mid = (points[lh, :2] + points[rh, :2]) / 2
    trunk = shoulder_mid - hip_mid
    out['trunk_lean_deg'] = math.degrees(math.atan2(trunk[0], -trunk[1]))
    for side in ('left', 'right'):
        pre = side.upper()
        hip = int(getattr(pose, f'{pre}_HIP'))
        knee = int(getattr(pose, f'{pre}_KNEE'))
        ankle = int(getattr(pose, f'{pre}_ANKLE'))
        out[f'{side}_fppa_deg'] = 180 - angle(points[hip, :2], points[knee, :2], points[ankle, :2])
        offset = float(points[knee, 0] - points[ankle, 0])
        out[f'{side}_knee_foot_offset_cm'] = offset / px_per_cm if px_per_cm else offset
    return out

def draw_stepdown_panel(frame, current, running, calibrated, lang='pt'):
    le = 'E' if lang != 'en' else 'L'
    ri = 'D' if lang != 'en' else 'R'
    labels = {'pelvic_tilt_deg': tr(lang, 'pelvis'), 'trunk_lean_deg': tr(lang, 'trunk'), 'left_fppa_deg': f'FPPA {le}', 'right_fppa_deg': f'FPPA {ri}', 'left_knee_foot_offset_cm': f"{tr(lang, 'kneefoot')} {le}", 'right_knee_foot_offset_cm': f"{tr(lang, 'kneefoot')} {ri}"}
    x0 = max(8, frame.shape[1] - 520)
    rows = []
    for key in labels:
        if key not in current:
            continue
        v = current[key]
        mn, mx = running[key]
        unit = 'cm' if key.endswith('_cm') and calibrated else 'px' if key.endswith('_cm') else 'deg'
        rows.append((labels[key], v, mn, mx, unit, (210, 160, 255) if 'fppa' in key else (80, 220, 255)))
    x_name = x0 + 14
    x_current = x0 + 250
    x_min = x0 + 330
    x_max = x0 + 400
    bottom = 134 + 32 * len(rows)
    dark_panel(frame, (x0, 58), (frame.shape[1] - 8, bottom))
    cv2.putText(frame, tr(lang, 'stepdown'), (x0 + 14, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (255, 255, 255), 2, cv2.LINE_AA)
    for text, x in ((tr(lang, 'measure_col'), x_name), (tr(lang, 'current'), x_current), (tr(lang, 'min'), x_min), (tr(lang, 'max'), x_max)):
        cv2.putText(frame, text, (x, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (220, 225, 235), 2, cv2.LINE_AA)
    for n, (name, v, mn, mx, unit, color) in enumerate(rows):
        y = 148 + 32 * n
        cv2.putText(frame, name, (x_name, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
        for value, x in ((v, x_current), (mn, x_min), (mx, x_max)):
            cv2.putText(frame, f'{value:.1f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
        cv2.putText(frame, unit, (x_max + 50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

def draw_right_panel(frame, phase_rows, steps, strides, lang='pt'):
    """Painel responsivo com fases atuais e historico completo das distancias."""
    height, width = frame.shape[:2]
    panel_w = min(520, max(390, int(width * 0.34)))
    x0 = width - panel_w - 8
    y0 = 58
    rows = max(len(steps), len(strides), 1)
    available = max(160, height - 210)
    row_h = max(18, min(30, available // rows))
    font = max(0.46, min(0.68, row_h / 42))
    bottom = min(height - 8, 190 + row_h * rows)
    dark_panel(frame, (x0, y0), (width - 8, bottom))
    cv2.putText(frame, tr(lang, 'gait'), (x0 + 16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
    for n, (label, phase, color) in enumerate(phase_rows):
        cv2.putText(frame, f'{label}: {phase}', (x0 + 16, 120 + 30 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.67, color, 2, cv2.LINE_AA)
    left = x0 + 16
    right = x0 + panel_w // 2 + 4
    cv2.putText(frame, tr(lang, 'steps'), (left, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, tr(lang, 'strides'), (right, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 120, 220), 2, cv2.LINE_AA)
    side_colors = {'E': (30, 230, 255), 'D': (255, 170, 30), 'L': (30, 230, 255), 'R': (255, 170, 30)}
    for n, (number, side, value) in enumerate(steps):
        cv2.putText(frame, f'{number:02d} ({side})  {value:6.1f} cm', (left, 211 + n * row_h), cv2.FONT_HERSHEY_SIMPLEX, font, side_colors[side], 2, cv2.LINE_AA)
    for n, (number, side, value) in enumerate(strides):
        cv2.putText(frame, f'{number:02d} ({side})  {value:6.1f} cm', (right, 211 + n * row_h), cv2.FONT_HERSHEY_SIMPLEX, font, side_colors[side], 2, cv2.LINE_AA)

def read_frame(data, index):
    cap = cv2.VideoCapture(str(data.input_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f'Nao foi possivel ler o quadro {index}.')
    return frame

def ear_center(points):
    """Centro das orelhas visiveis; usa uma so quando a outra esta oclusa."""
    if points is None:
        return None
    ids = [int(mp.solutions.pose.PoseLandmark.LEFT_EAR), int(mp.solutions.pose.PoseLandmark.RIGHT_EAR)]
    visible = [points[i, :2] for i in ids if points[i, 2] >= 0.35]
    return np.mean(visible, axis=0) if visible else None

def draw_overlay(frame, points, opts, calibration=None, editable=False, ear_trail=None):
    frame = frame.copy()
    metrics = {}
    if points is None:
        cv2.putText(frame, 'Pessoa nao detectada', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 30, 230), 3)
        return (frame, metrics)
    pose = mp.solutions.pose.PoseLandmark
    for a, b in mp.solutions.pose.POSE_CONNECTIONS:
        if points[a, 2] >= opts.visibility_threshold and points[b, 2] >= opts.visibility_threshold:
            cv2.line(frame, tuple(points[a, :2].astype(int)), tuple(points[b, :2].astype(int)), (255, 180, 30), 3)
    for p in points:
        if p[2] >= opts.visibility_threshold:
            cv2.circle(frame, tuple(p[:2].astype(int)), 7 if editable else 5, (60, 230, 60), -1)
    scores = {s: np.mean([points[int(getattr(pose, f'{s.upper()}_{j}')), 2] for j in ('HIP', 'KNEE', 'ANKLE')]) for s in ('left', 'right')}
    sides = {'Esquerdo': ['left'], 'Direito': ['right'], 'Ambos': ['left', 'right']}.get(opts.side) or [max(scores, key=scores.get)]
    for side in sides:
        ids = {n: int(getattr(pose, f'{side.upper()}_{n}')) for n in ('SHOULDER', 'HIP', 'KNEE', 'ANKLE', 'FOOT_INDEX')}
        for joint, names in {'hip': ('SHOULDER', 'HIP', 'KNEE'), 'knee': ('HIP', 'KNEE', 'ANKLE'), 'ankle': ('KNEE', 'ANKLE', 'FOOT_INDEX')}.items():
            ix = [ids[n] for n in names]
            if min((points[k, 2] for k in ix)) < opts.visibility_threshold:
                continue
            value = angle(points[ix[0], :2], points[ix[1], :2], points[ix[2], :2])
            metrics[f'{side}_{joint}_deg'] = value
            x, y = points[ix[1], :2].astype(int)
            color = (30, 230, 255) if side == 'left' else (255, 170, 30)
            side_letter = ('L' if side == 'left' else 'R') if opts.language == 'en' else 'E' if side == 'left' else 'D'
            label = f'{side_letter} {tr(opts.language, joint)}: {value:.1f} deg'
            cv2.putText(frame, label, (x + 10, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(frame, label, (x + 10, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.92, color, 3, cv2.LINE_AA)
        if opts.view == 'Frontal':
            hip, knee, ankle = (ids['HIP'], ids['KNEE'], ids['ANKLE'])
            if min((points[k, 2] for k in (hip, knee, ankle))) >= opts.visibility_threshold:
                q_value = 180.0 - angle(points[hip, :2], points[knee, :2], points[ankle, :2])
                metrics[f'{side}_q_estimated_deg'] = q_value
                x, y = points[knee, :2].astype(int)
                side_letter = ('L' if side == 'left' else 'R') if opts.language == 'en' else 'E' if side == 'left' else 'D'
                label = f"{tr(opts.language, 'q')} {side_letter}: {q_value:.1f} deg"
                cv2.putText(frame, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(frame, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (210, 80, 255), 3, cv2.LINE_AA)
    if opts.view == 'Frontal':
        for q_side in ('left', 'right'):
            key = f'{q_side}_q_estimated_deg'
            if key in metrics:
                continue
            pre = q_side.upper()
            hip = int(getattr(pose, f'{pre}_HIP'))
            knee = int(getattr(pose, f'{pre}_KNEE'))
            ankle = int(getattr(pose, f'{pre}_ANKLE'))
            if min((points[k, 2] for k in (hip, knee, ankle))) >= opts.visibility_threshold:
                q_value = 180.0 - angle(points[hip, :2], points[knee, :2], points[ankle, :2])
                metrics[key] = q_value
                x, y = points[knee, :2].astype(int)
                side_letter = ('L' if q_side == 'left' else 'R') if opts.language == 'en' else 'E' if q_side == 'left' else 'D'
                label = f"{tr(opts.language, 'q')} {side_letter}: {q_value:.1f} deg"
                cv2.putText(frame, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(frame, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (210, 80, 255), 3, cv2.LINE_AA)
    if calibration:
        a, b = (tuple(map(int, p)) for p in calibration)
        cv2.line(frame, a, b, (220, 40, 220), 4)
        cv2.circle(frame, a, 8, (220, 40, 220), -1)
        cv2.circle(frame, b, 8, (220, 40, 220), -1)
    if opts.view == 'Lateral' and ear_trail:
        trail = [tuple(np.asarray(p).astype(int)) for p in ear_trail if p is not None]
        if len(trail) > 1:
            cv2.polylines(frame, [np.asarray(trail, dtype=np.int32)], False, (30, 80, 255), 4, cv2.LINE_AA)
        center = ear_center(points)
        if center is not None:
            cv2.circle(frame, tuple(center.astype(int)), 9, (30, 80, 255), -1)
            cv2.circle(frame, tuple(center.astype(int)), 12, (255, 255, 255), 2)
    return (frame, metrics)

def draw_manual_segments(frame, data, lang='pt', points=None):
    if not data.manual_segments and (not data.tracked_segments) or not data.px_per_cm:
        return frame
    values = []
    for segment in data.manual_segments:
        color = tuple(segment.get('color', (70, 255, 230)))
        a = np.asarray(segment['p1'])
        b = np.asarray(segment['p2'])
        value = float(np.linalg.norm(a - b) / data.px_per_cm)
        values.append((segment['name'], value, color))
        pa, pb = (tuple(a.astype(int)), tuple(b.astype(int)))
        cv2.line(frame, pa, pb, color, 4, cv2.LINE_AA)
        cv2.circle(frame, pa, 7, color, -1)
        cv2.circle(frame, pb, 7, color, -1)
        mid = tuple(((a + b) / 2).astype(int))
        cv2.putText(frame, f"{segment['name']}: {value:.1f} cm", (mid[0] + 8, mid[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, f"{segment['name']}: {value:.1f} cm", (mid[0] + 8, mid[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if points is not None:
        for segment in data.tracked_segments:
            color = tuple(segment.get('color', (255, 210, 60)))
            a = points[segment['i1'], :2]
            b = points[segment['i2'], :2]
            value = float(np.linalg.norm(a - b) / data.px_per_cm)
            values.append((segment['name'], value, color))
            pa, pb = (tuple(a.astype(int)), tuple(b.astype(int)))
            cv2.line(frame, pa, pb, color, 4, cv2.LINE_AA)
            cv2.circle(frame, pa, 8, color, -1)
            cv2.circle(frame, pb, 8, color, -1)
            mid = tuple(((a + b) / 2).astype(int))
            label = f"{segment['name']}: {value:.1f} cm"
            cv2.putText(frame, label, (mid[0] + 8, mid[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, label, (mid[0] + 8, mid[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if not values:
        return frame
    row_h = 28
    box_h = 50 + row_h * len(values)
    y0 = max(58, frame.shape[0] - box_h - 8)
    title = tr(lang, 'manual')
    width = max([text_width(f'{n}: {v:.1f} cm', 0.62, 2) for n, v, _ in values] + [text_width(title, 0.68, 2)]) + 36
    dark_panel(frame, (8, y0), (min(frame.shape[1] - 8, 8 + width), frame.shape[0] - 8))
    cv2.putText(frame, title, (20, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    for i, (name, value, color) in enumerate(values):
        cv2.putText(frame, f'{name}: {value:.1f} cm', (20, y0 + 56 + i * row_h), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return frame

def _render_video(data, out_path, opts, progress=None):
    cap = cv2.VideoCapture(str(data.input_path))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), data.fps, (data.width, data.height))
    if not writer.isOpened():
        raise ValueError('Nao foi possivel criar o video de saida.')
    if opts.side == 'Automatico':
        opts = ProcessingOptions(opts.view, 'Esquerdo' if data.predominant_side == 'left' else 'Direito', opts.visibility_threshold, opts.language)
    phases = gait_phases(data) if opts.view == 'Lateral' else None
    ear_layer = np.zeros((data.height, data.width, 3), np.uint8)
    ear_mask = np.zeros((data.height, data.width), np.uint8)
    rows = []
    total = len(data.landmarks)
    trail = []
    first_ear = None
    previous_ear = None
    ear_path_px = 0.0
    running = {}
    stepdown_running = {}
    last_strike = {'left': None, 'right': None}
    last_contact_frame = {'left': -10 ** 9, 'right': -10 ** 9}
    last_accepted_side = None
    last_accepted_frame = -10 ** 9
    last_footfall_x = None
    latest_step = None
    latest_stride = {'left': None, 'right': None}
    step_history = []
    stride_history = []
    footfall_positions = []
    for i, points in enumerate(data.landmarks):
        ok, frame = cap.read()
        if not ok:
            break
        current_ear = ear_center(points) if opts.view == 'Lateral' else None
        if current_ear is not None:
            trail.append(current_ear.copy())
            first_ear = current_ear.copy() if first_ear is None else first_ear
            if previous_ear is not None:
                ear_path_px += float(np.linalg.norm(current_ear - previous_ear))
                a, b = (tuple(previous_ear.astype(int)), tuple(current_ear.astype(int)))
                cv2.line(ear_layer, a, b, (30, 80, 255), 4, cv2.LINE_AA)
                cv2.line(ear_mask, a, b, 255, 5, cv2.LINE_AA)
            previous_ear = current_ear.copy()
        frame, metrics = draw_overlay(frame, points, opts, data.calibration_points, ear_trail=None)
        if current_ear is not None:
            cv2.copyTo(ear_layer, ear_mask, frame)
            center = tuple(current_ear.astype(int))
            cv2.circle(frame, center, 9, (30, 80, 255), -1)
            cv2.circle(frame, center, 12, (255, 255, 255), 2)
        cv2.rectangle(frame, (0, 0), (data.width, 52), (20, 20, 20), -1)
        view_label = tr(opts.language, 'lateral' if opts.view == 'Lateral' else 'frontal')
        cv2.putText(frame, f"{tr(opts.language, 'analysis')} | {view_label} | t={i / data.fps:.2f}s", (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 3)
        sd_metrics = stepdown_metrics(points, data.px_per_cm) if opts.view == 'Frontal' else {}
        for key, value in metrics.items():
            if key.endswith('_deg') and np.isfinite(value):
                running[key] = (min(running.get(key, (value, value))[0], value), max(running.get(key, (value, value))[1], value))
        draw_angle_table(frame, metrics, running, opts.language)
        for key, value in sd_metrics.items():
            if np.isfinite(value):
                stepdown_running[key] = (min(stepdown_running.get(key, (value, value))[0], value), max(stepdown_running.get(key, (value, value))[1], value))
        if sd_metrics:
            draw_stepdown_panel(frame, sd_metrics, stepdown_running, data.px_per_cm is not None, opts.language)
        row = {'frame': i, 'time_s': i / data.fps, **metrics, **sd_metrics}
        if points is not None and data.px_per_cm:
            la, ra = (int(mp.solutions.pose.PoseLandmark.LEFT_ANKLE), int(mp.solutions.pose.PoseLandmark.RIGHT_ANKLE))
            delta = points[la, :2] - points[ra, :2]
            row['ankles_distance_cm'] = float(np.linalg.norm(delta) / data.px_per_cm)
            row['ankles_horizontal_cm'] = float(abs(delta[0]) / data.px_per_cm)
            if current_ear is not None:
                row['ear_displacement_cm'] = float(np.linalg.norm(current_ear - first_ear) / data.px_per_cm)
                row['ear_path_cm'] = ear_path_px / data.px_per_cm
        if phases and points is not None:
            pose = mp.solutions.pose.PoseLandmark
            phase_rows = []
            for n, side in enumerate(('left', 'right')):
                stance = bool(phases[side]['stance'][i])
                event = phases[side]['event'][i]
                row[f'{side}_phase'] = 'APOIO' if stance else 'BALANCO'
                row[f'{side}_event'] = event
                foot_color = (40, 220, 80) if stance else (30, 170, 255)
                heel = int(getattr(pose, f'{side.upper()}_HEEL'))
                toe = int(getattr(pose, f'{side.upper()}_FOOT_INDEX'))
                ankle = int(getattr(pose, f'{side.upper()}_ANKLE'))
                cv2.line(frame, tuple(points[heel, :2].astype(int)), tuple(points[toe, :2].astype(int)), foot_color, 7, cv2.LINE_AA)
                side_letter = ('L' if side == 'left' else 'R') if opts.language == 'en' else 'E' if side == 'left' else 'D'
                phase_name = tr(opts.language, 'support' if stance else 'swing')
                phase_rows.append((side_letter, phase_name, foot_color))
                if event:
                    phase_rows[-1] = (side_letter, f"{phase_name} - {tr(opts.language, 'contact' if event == 'CONTATO INICIAL' else 'toeoff')}", (40, 255, 255))
                if event == 'CONTATO INICIAL' and data.px_per_cm and (i - last_contact_frame[side] >= max(6, int(data.fps * 0.35))):
                    other = int(getattr(pose, f"{('RIGHT' if side == 'left' else 'LEFT')}_ANKLE"))
                    candidate_step = float(abs(points[ankle, 0] - points[other, 0]) / data.px_per_cm)
                    heel_x = float(points[heel, 0])
                    spatial_ok = last_footfall_x is None or abs(heel_x - last_footfall_x) / data.px_per_cm >= 5.0
                    sequence_ok = last_accepted_side is None or side != last_accepted_side
                    time_ok = i - last_accepted_frame >= max(6, int(data.fps * 0.25))
                    if candidate_step >= 5.0 and spatial_ok and sequence_ok and time_ok:
                        last_contact_frame[side] = i
                        last_accepted_frame = i
                        last_accepted_side = side
                        last_footfall_x = heel_x
                        latest_step = candidate_step
                        step_history.append((len(step_history) + 1, side_letter, latest_step))
                        row['step_number'] = len(step_history)
                        row['step_side'] = side_letter
                        footfall_positions.append((len(step_history), side_letter, points[heel, :2].copy()))
                        candidate_stride = None if last_strike[side] is None else abs(heel_x - last_strike[side]) / data.px_per_cm
                        last_strike[side] = heel_x
                        if candidate_stride is not None and candidate_stride >= 5.0:
                            latest_stride[side] = candidate_stride
                            stride_history.append((len(stride_history) + 1, side_letter, candidate_stride))
                            row['stride_number'] = len(stride_history)
                            row['stride_side'] = side_letter
                        cv2.line(frame, tuple(points[ankle, :2].astype(int)), tuple(points[other, :2].astype(int)), (255, 60, 220), 4)
            if latest_step is not None:
                row['step_length_cm'] = latest_step
            strides = [v for v in latest_stride.values() if v is not None]
            if strides:
                row['stride_length_cm'] = strides[-1]
            for number, side_letter, position in footfall_positions:
                p = tuple(position.astype(int))
                color = (30, 230, 255) if side_letter == 'E' else (255, 170, 30)
                cv2.circle(frame, p, 13, color, 3, cv2.LINE_AA)
                cv2.putText(frame, str(number), (p[0] - 5, p[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            draw_right_panel(frame, phase_rows, step_history, stride_history, opts.language)
        draw_manual_segments(frame, data, opts.language, points)
        rows.append(row)
        writer.write(frame)
        if progress and i % 5 == 0:
            progress(i * 100 / total, f"{tr(opts.language, 'export_frame')} {i + 1} {tr(opts.language, 'of')} {total}...")
    cap.release()
    writer.release()
    csv_path = out_path.with_suffix('.csv')
    fields = ['frame', 'time_s', 'left_hip_deg', 'left_knee_deg', 'left_ankle_deg', 'right_hip_deg', 'right_knee_deg', 'right_ankle_deg', 'left_q_estimated_deg', 'right_q_estimated_deg', 'pelvic_tilt_deg', 'trunk_lean_deg', 'left_fppa_deg', 'right_fppa_deg', 'left_knee_foot_offset_cm', 'right_knee_foot_offset_cm', 'left_phase', 'right_phase', 'left_event', 'right_event', 'step_number', 'step_side', 'step_length_cm', 'stride_number', 'stride_side', 'stride_length_cm', 'ankles_distance_cm', 'ankles_horizontal_cm', 'ear_displacement_cm', 'ear_path_cm']
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return csv_path

class VideoExporter:
    """Coordena a geracao do MP4 analisado e de seu arquivo CSV."""

    def __init__(self, options, progress=None):
        self.options = options
        self.progress = progress

    def export(self, analysis, output_path):
        return _render_video(analysis, output_path, self.options, self.progress)

def render_video(data, out_path, opts, progress=None):
    """Fachada publica preservada para nao alterar o codigo da interface."""
    return VideoExporter(opts, progress).export(data, out_path)
