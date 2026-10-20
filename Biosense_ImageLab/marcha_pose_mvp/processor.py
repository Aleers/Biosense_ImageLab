from __future__ import annotations
import math
from datetime import datetime, timezone
from uuid import uuid4
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable
import cv2, mediapipe as mp, numpy as np
from i18n import tr
from joint_angles import JointAngleCalculator, JOINT_DEFINITIONS, angle_2d
from csv_exporter import CsvExporter
from landmark_quality import LandmarkValidator
from session import SessionStore, json_safe
from temporal_filter import TemporalLandmarkFilter
from artifact_io import ArtifactPublisher
from gait_events import GaitPhaseAnalyzer, FootfallRegistry
from visual_continuity import VisualContinuity, draw_joint_labels, dashed_line

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
    smoothing_seconds: float = 0.0
    tracking_use_optical_flow: bool = False
    tracking_smoothing_seconds: float = 0.0
    visual_hold_seconds: float = 0.20

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

    raw_landmarks: tuple = field(default_factory=tuple)
    timestamps: list[float] = field(default_factory=list)
    tracking_records: list[dict] = field(default_factory=list)
    gait_records: list[dict] = field(default_factory=list)
    display_records: list[dict] = field(default_factory=list)
    review_history: list[dict] = field(default_factory=list)
    audit: list[dict] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: uuid4().hex)
    session_path: Path | None = None
    source_identity: dict = field(default_factory=dict)
    software_identity: dict = field(default_factory=dict)
    saved_options: dict = field(default_factory=dict)
    full_precision_rows: list[dict] = field(default_factory=list)

    def capture_originals(self, sequence=None):
        if self.raw_landmarks:
            return
        originals = []
        for points in self.landmarks if sequence is None else sequence:
            if points is None:
                originals.append(None)
            else:
                original = points.copy()
                original.setflags(write=False)
                originals.append(original)
        self.raw_landmarks = tuple(originals)
        if not self.timestamps:
            self.timestamps = [i / self.fps for i in range(len(self.landmarks))]

    def record(self, action, **details):
        event = json_safe(dict(id=len(self.audit)+1, action=action,
                         timestamp=datetime.now(timezone.utc).isoformat(), **details))
        self.audit.append(event)
        if action in ('tracking_result', 'review_action', 'undo', 'temporal_filter', 'side_identity_correction'):
            self.full_precision_rows = []
            self.gait_records = []
            self.display_records = []
        return event['id']

    @property
    def px_per_cm(self):
        return LandmarkValidator.scale(self.calibration_points, self.calibration_cm)

def angle(a, b, c):
    return angle_2d(a, b, c)

class PoseVideoAnalyzer:
    """Le o video e converte cada quadro em landmarks do MediaPipe.

    Esta classe possui uma unica responsabilidade: executar a deteccao. As
    correcoes manuais e a exportacao permanecem em etapas separadas.
    """

    def __init__(self, language='pt', progress=None, smoothing_seconds=0.0):
        self.language = language
        self.progress = progress
        self.smoothing_seconds = smoothing_seconds

    def analyze(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise ValueError('Nao foi possivel abrir o video.')
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError('FPS invalido; nao e possivel garantir a escala temporal.')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise ValueError('Dimensoes invalidas do video.')
            reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total = max(1, reported_total)
            data = Analysis(Path(path), fps, width, height)
            pose = mp.solutions.pose
            with pose.Pose(model_complexity=2, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
                frame_number = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        if reported_total > 0 and frame_number < reported_total:
                            raise ValueError(f'Leitura incompleta: quadro {frame_number} de {reported_total}.')
                        break
                    found = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks
                    points = None
                    if found:
                        points = np.array([[p.x*width, p.y*height, p.visibility] for p in found.landmark], dtype=np.float32)
                    data.landmarks.append(points)
                    frame_number += 1
                    self._report_progress(frame_number, total)
        finally:
            cap.release()
        if not data.landmarks:
            raise ValueError('O video nao contem quadros validos.')
        data.capture_originals()
        data.record('detection', frames=len(data.landmarks), model='mediapipe_pose', model_complexity=2,
                    builtin_smoothing=True, timestamp_source='nominal_fps')
        stabilize_lower_limbs(data)
        if self.smoothing_seconds > 0:
            smoother = TemporalLandmarkFilter(self.smoothing_seconds)
            data.landmarks = [smoother.update(points, timestamp) for points, timestamp in zip(data.landmarks, data.timestamps)]
            data.record('temporal_filter', time_constant_seconds=self.smoothing_seconds)
        return data

    def _report_progress(self, frame_number, total):
        if self.progress and frame_number % 5 == 0:
            message = f"{tr(self.language, 'detecting')}: {frame_number} {tr(self.language, 'of')} {total}..."
            self.progress(min(99, frame_number * 100 / total), message)

def analyze_video(path: Path, progress: Callable | None=None, language='pt', smoothing_seconds=0.0):
    """Fachada mantida para compatibilidade com a interface existente."""
    return PoseVideoAnalyzer(language, progress, smoothing_seconds).analyze(path)

def stabilize_lower_limbs(data):
    """Mantem a identidade esquerda/direita e fixa o lado predominante do video."""
    data.capture_originals()
    pose = mp.solutions.pose.PoseLandmark
    names = ('HIP', 'KNEE', 'ANKLE', 'HEEL', 'FOOT_INDEX')
    pairs = [(int(getattr(pose, f'LEFT_{n}')), int(getattr(pose, f'RIGHT_{n}'))) for n in names]
    scores = {'left': [], 'right': []}
    previous = None
    for frame_index, points in enumerate(data.landmarks):
        if points is None:
            previous = None
            continue
        if previous is not None and LandmarkValidator().valid(points, tuple(i for pair in pairs for i in pair)) and LandmarkValidator().valid(previous, tuple(i for pair in pairs for i in pair)):
            normal = sum((np.linalg.norm(points[a, :2] - previous[a, :2]) + np.linalg.norm(points[b, :2] - previous[b, :2]) for a, b in pairs))
            swapped = sum((np.linalg.norm(points[b, :2] - previous[a, :2]) + np.linalg.norm(points[a, :2] - previous[b, :2]) for a, b in pairs))
            if swapped < normal * 0.72:
                swapped_pairs = list(pairs)
                ls, rs = int(pose.LEFT_SHOULDER), int(pose.RIGHT_SHOULDER)
                if LandmarkValidator().valid(points, (ls, rs)) and LandmarkValidator().valid(previous, (ls, rs)):
                    direct = np.linalg.norm(points[ls, :2]-previous[ls, :2])+np.linalg.norm(points[rs, :2]-previous[rs, :2])
                    reverse = np.linalg.norm(points[rs, :2]-previous[ls, :2])+np.linalg.norm(points[ls, :2]-previous[rs, :2])
                    if reverse < direct*0.72:
                        swapped_pairs.append((ls, rs))
                for a, b in swapped_pairs:
                    points[[a, b]] = points[[b, a]]
                data.record('side_identity_correction', frame=frame_index, pairs=swapped_pairs)
            for a, b in pairs:
                for idx in (a, b):
                    if points[idx, 2] < 0.42 and previous[idx, 2] >= 0.42:
                        points[idx, :2] = previous[idx, :2]
                        points[idx, 2] = min(points[idx, 2], 0.42)
        scores['left'].append(float(np.nanmean([points[a, 2] for a, _ in pairs])))
        scores['right'].append(float(np.nanmean([points[b, 2] for _, b in pairs])))
        previous = points.copy()
    data.predominant_side = max(scores, key=lambda s: np.median(scores[s]) if scores[s] else 0)

def gait_phases(data, visibility_threshold=0.45):
    """Compatibility facade for time-based phase confirmation."""
    return GaitPhaseAnalyzer(visibility_threshold).analyze(data)


def draw_angle_table(frame, current, running, lang='pt', status=None):
    status = status or {}
    keys = [k for k in current if k.endswith('_deg') and any((tag in k for tag in ('_hip_', '_knee_', '_ankle_', '_q_estimated_')))]
    if not keys:
        return
    labels = {'left': 'E' if lang != 'en' else 'L', 'right': 'D' if lang != 'en' else 'R', 'hip': tr(lang, 'hip'), 'knee': tr(lang, 'knee'), 'ankle': tr(lang, 'ankle'), 'q': tr(lang, 'q')}
    rows = []
    for key in sorted(keys):
        parts = key.split('_')
        name = f'{labels.get(parts[0], parts[0])} {labels.get(parts[1], parts[1])}'
        v = current[key]
        mn, mx = running.get(key, (None, None))
        rows.append((name, v, mn, mx, status.get(key)))
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
    for row, (name, v, mn, mx, state) in enumerate(rows):
        y = 148 + 32 * row
        cv2.putText(frame, name, (x_name, y), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (235, 242, 255), 2, cv2.LINE_AA)
        for value, x in ((v, x_current), (mn, x_min), (mx, x_max)):
            label = '--' if value is None or not np.isfinite(value) else f'{value:.1f}'
            if x == x_current and state == 'held':
                label += '~'
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (235, 242, 255), 2, cv2.LINE_AA)

def stepdown_metrics(points, px_per_cm=None, visibility_threshold=0.45):
    validator = LandmarkValidator(visibility_threshold)
    out = {}
    if validator.valid(points, (23, 24)):
        pelvis = points[24, :2]-points[23, :2]
        if np.linalg.norm(pelvis) > 1e-8:
            out['pelvic_tilt_deg'] = math.degrees(math.atan2(pelvis[1], pelvis[0]))
    if validator.valid(points, (11, 12, 23, 24)):
        trunk = (points[11, :2]+points[12, :2]-points[23, :2]-points[24, :2])/2
        if np.linalg.norm(trunk) > 1e-8:
            out['trunk_lean_deg'] = math.degrees(math.atan2(trunk[0], -trunk[1]))
    for side, hip, knee, ankle in [('left', 23, 25, 27), ('right', 24, 26, 28)]:
        if validator.valid(points, (hip, knee, ankle)):
            value = angle(points[hip, :2], points[knee, :2], points[ankle, :2])
            if np.isfinite(value):
                out[f'{side}_fppa_deg'] = 180-value
        if validator.valid(points, (knee, ankle)):
            offset = float(points[knee, 0]-points[ankle, 0])
            out[f'{side}_knee_foot_offset_cm'] = offset/px_per_cm if px_per_cm else offset
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
        if key not in running or not np.isfinite(v):
            continue
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
        value_label = '--' if value is None else f'{value:6.1f}'
        cv2.putText(frame, f'{number:02d} ({side})  {value_label} cm', (left, 211 + n * row_h), cv2.FONT_HERSHEY_SIMPLEX, font, side_colors[side], 2, cv2.LINE_AA)
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
    visible = [points[i, :2] for i in ids if LandmarkValidator(0.35).valid(points, (i,))]
    return np.mean(visible, axis=0) if visible else None

def draw_overlay(frame, points, opts, calibration=None, editable=False, ear_trail=None, display=None, draw_labels=True):
    frame = frame.copy()
    pose = mp.solutions.pose.PoseLandmark
    if display is not None:
        sides=display.sides
    else:
        scores={side: np.mean([points[int(getattr(pose,f'{side.upper()}_{joint}')),2] for joint in ('HIP','KNEE','ANKLE')])
                if points is not None else 0 for side in ('left','right')}
        sides={'Esquerdo':('left',),'Direito':('right',),'Ambos':('left','right')}.get(opts.side) or (max(scores,key=scores.get),)
    calculator=JointAngleCalculator(opts.visibility_threshold)
    metrics=calculator.calculate(points,sides)
    if opts.view=='Frontal':
        both=calculator.calculate(points,('left','right'))
        for side in ('left','right'):
            if f'{side}_knee_deg' in both:
                metrics[f'{side}_q_estimated_deg']=180-both[f'{side}_knee_deg']
    rendered=display.points if display is not None else points
    if rendered is not None:
        for a,b in mp.solutions.pose.POSE_CONNECTIONS:
            if not np.all(np.isfinite(rendered[[a,b],:2])):
                continue
            if display is not None and (display.landmark_status[a]!='measured' or display.landmark_status[b]!='measured'):
                color=(30,170,255) if 'held' in (display.landmark_status[a],display.landmark_status[b]) else (140,140,140)
                dashed_line(frame,rendered[a,:2],rendered[b,:2],color)
            elif LandmarkValidator(opts.visibility_threshold).valid(rendered,(a,b)):
                cv2.line(frame,tuple(rendered[a,:2].astype(int)),tuple(rendered[b,:2].astype(int)),(255,180,30),3)
        for index,point in enumerate(rendered):
            if not np.all(np.isfinite(point[:2])):
                continue
            state=display.landmark_status[index] if display is not None else 'measured'
            if display is None and not LandmarkValidator(opts.visibility_threshold).valid(rendered,(index,)):
                continue
            color=(60,230,60) if state=='measured' else (30,170,255) if state=='held' else (140,140,140)
            cv2.circle(frame,tuple(point[:2].astype(int)),7 if editable else 5,color,-1 if state=='measured' else 2)
    if draw_labels:
        draw_joint_labels(frame,rendered,display.metrics if display is not None else metrics,opts,
                          display.status if display is not None else None)
    if points is None:
        cv2.putText(frame,tr(opts.language,'signal_lost'),(20,45),cv2.FONT_HERSHEY_SIMPLEX,.7,(30,30,230),2)
    if calibration:
        a,b=(tuple(map(int,p)) for p in calibration)
        cv2.line(frame,a,b,(220,40,220),4)
        cv2.circle(frame,a,8,(220,40,220),-1)
        cv2.circle(frame,b,8,(220,40,220),-1)
    if opts.view=='Lateral' and ear_trail:
        trail=[tuple(np.asarray(p).astype(int)) for p in ear_trail if p is not None]
        if len(trail)>1:
            cv2.polylines(frame,[np.asarray(trail,dtype=np.int32)],False,(30,80,255),4,cv2.LINE_AA)
        center=ear_center(points)
        if center is not None:
            cv2.circle(frame,tuple(center.astype(int)),9,(30,80,255),-1)
    return frame,metrics


def draw_manual_segments(frame, data, lang='pt', points=None, visibility_threshold=0.45):
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
            if not LandmarkValidator(visibility_threshold).valid(points, (segment['i1'], segment['i2'])):
                continue
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

def _render_video(data, out_path, opts, progress=None, published_output=None):
    out_path = Path(out_path)
    if out_path.resolve() == data.input_path.resolve():
        raise ValueError('A saida nao pode sobrescrever o video original.')
    cap = cv2.VideoCapture(str(data.input_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError('Nao foi possivel reabrir o video original.')
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), data.fps, (data.width, data.height))
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise ValueError('Nao foi possivel criar o video de saida.')
    try:
        if opts.side == 'Automatico':
            opts = replace(opts, side='Esquerdo' if data.predominant_side == 'left' else 'Direito')
        phases = gait_phases(data, opts.visibility_threshold) if opts.view == 'Lateral' else None
        ear_layer = np.zeros((data.height, data.width, 3), np.uint8)
        ear_mask = np.zeros((data.height, data.width), np.uint8)
        rows = []
        total = len(data.landmarks)
        visual = VisualContinuity(opts)
        display_frames = visual.build(data.landmarks, data.fps)
        data.display_records = visual.records
        registry = FootfallRegistry(data, opts.visibility_threshold)
        trail = []
        first_ear = None
        previous_ear = None
        ear_path_px = 0.0
        running = {}
        stepdown_running = {}
        for i, points in enumerate(data.landmarks):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Exportacao incompleta: quadro {i} nao pode ser lido.')
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
            frame, metrics = draw_overlay(frame, points, opts, data.calibration_points, ear_trail=None, display=display_frames[i], draw_labels=False)
            if previous_ear is not None:
                cv2.copyTo(ear_layer, ear_mask, frame)
            if current_ear is not None:
                center = tuple(current_ear.astype(int))
                cv2.circle(frame, center, 9, (30, 80, 255), -1)
                cv2.circle(frame, center, 12, (255, 255, 255), 2)
            cv2.rectangle(frame, (0, 0), (data.width, 52), (20, 20, 20), -1)
            view_label = tr(opts.language, 'lateral' if opts.view == 'Lateral' else 'frontal')
            cv2.putText(frame, f"{tr(opts.language, 'analysis')} | {view_label} | t={i / data.fps:.2f}s", (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 3)
            sd_metrics = stepdown_metrics(points, data.px_per_cm, opts.visibility_threshold) if opts.view == 'Frontal' else {}
            for key, value in metrics.items():
                if key.endswith('_deg') and np.isfinite(value):
                    running[key] = (min(running.get(key, (value, value))[0], value), max(running.get(key, (value, value))[1], value))
            draw_angle_table(frame, display_frames[i].metrics, running, opts.language, display_frames[i].status)
            for key, value in sd_metrics.items():
                if np.isfinite(value):
                    stepdown_running[key] = (min(stepdown_running.get(key, (value, value))[0], value), max(stepdown_running.get(key, (value, value))[1], value))
            if sd_metrics:
                draw_stepdown_panel(frame, sd_metrics, stepdown_running, data.px_per_cm is not None, opts.language)
            row = {'frame': i, 'time_s': i / data.fps, **metrics, **sd_metrics}
            if points is not None and data.px_per_cm:
                la, ra = (int(mp.solutions.pose.PoseLandmark.LEFT_ANKLE), int(mp.solutions.pose.PoseLandmark.RIGHT_ANKLE))
                delta = points[la, :2] - points[ra, :2]
                validator = LandmarkValidator(opts.visibility_threshold)
                if validator.valid(points, (la, ra)):
                    row['ankles_distance_cm'] = validator.distance(points, (la, ra), data.px_per_cm)
                    row['ankles_horizontal_cm'] = validator.distance(points, (la, ra), data.px_per_cm, horizontal=True)
                if current_ear is not None:
                    row['ear_displacement_cm'] = float(np.linalg.norm(current_ear - first_ear) / data.px_per_cm)
                    row['ear_path_cm'] = ear_path_px / data.px_per_cm
            if phases:
                registration=registry.update(i,points,phases)
                letter=lambda side: ('L' if side=='left' else 'R') if opts.language=='en' else ('E' if side=='left' else 'D')
                for key in ('step_side','stride_side'):
                    if key in registration:
                        registration[key]=letter(registration[key])
                row.update(registration)
                phase_rows=[]
                for side,heel,toe in [('left',29,31),('right',30,32)]:
                    valid=bool(phases[side]['valid'][i])
                    stance=bool(phases[side]['stance'][i])
                    event=phases[side]['event'][i]
                    if valid:
                        row[f'{side}_phase']='APOIO' if stance else 'BALANCO'
                        row[f'{side}_event']=event
                    confirmed=stance and registry.support_confirmed[side]
                    color=(40,220,80) if confirmed else (30,220,255) if stance else (30,170,255)
                    if not valid:
                        label=tr(opts.language,'signal_lost')
                        color=(160,160,160)
                    elif stance and not confirmed:
                        label=tr(opts.language,'unconfirmed_support')
                    else:
                        label=tr(opts.language,'support' if stance else 'swing')
                    phase_rows.append((letter(side),label,color))
                    if LandmarkValidator(opts.visibility_threshold).valid(points,(heel,toe)):
                        cv2.line(frame,tuple(points[heel,:2].astype(int)),tuple(points[toe,:2].astype(int)),color,7,cv2.LINE_AA)
                steps=[(entry['number'],letter(entry['side']),entry['step_cm']) for entry in registry.footfalls]
                strides=[(entry['number'],letter(entry['side']),entry['length_cm']) for entry in registry.strides]
                for entry in registry.footfalls:
                    if entry['position'] is None:
                        continue
                    position=tuple(entry['position'].astype(int))
                    color=(30,230,255) if entry['side']=='left' else (255,170,30)
                    cv2.circle(frame,position,13,color,3,cv2.LINE_AA)
                    cv2.putText(frame,str(entry['number']),(position[0]-5,position[1]-18),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
                draw_right_panel(frame,phase_rows,steps,strides,opts.language)
            draw_manual_segments(frame, data, opts.language, points, opts.visibility_threshold)
            draw_joint_labels(frame, display_frames[i].points, display_frames[i].metrics, opts, display_frames[i].status)
            rows.append(row)
            writer.write(frame)
            if progress and i % 5 == 0:
                progress(i * 100 / total, f"{tr(opts.language, 'export_frame')} {i + 1} {tr(opts.language, 'of')} {total}...")
    finally:
        cap.release()
        writer.release()
    csv_path = out_path.with_suffix('.csv')
    fields = ['frame', 'time_s', 'left_hip_deg', 'left_knee_deg', 'left_ankle_deg', 'right_hip_deg', 'right_knee_deg', 'right_ankle_deg', 'left_q_estimated_deg', 'right_q_estimated_deg', 'pelvic_tilt_deg', 'trunk_lean_deg', 'left_fppa_deg', 'right_fppa_deg', 'left_knee_foot_offset_cm', 'right_knee_foot_offset_cm', 'left_phase', 'right_phase', 'left_event', 'right_event', 'step_number', 'step_side', 'step_length_cm', 'stride_number', 'stride_side', 'stride_length_cm', 'ankles_distance_cm', 'ankles_horizontal_cm', 'ear_displacement_cm', 'ear_path_cm']
    CsvExporter().export(csv_path, fields, rows)
    data.full_precision_rows = rows
    data.gait_records = registry.decisions + [dict(kind='footfall', **entry) for entry in registry.footfalls] if phases else []
    data.record('export_prepared', output=str(published_output or out_path), frames=len(rows))
    session_path = data.session_path or out_path.with_name(f'{out_path.stem}_{data.session_id[:8]}.session.zip')
    SessionStore().save(data, session_path, opts)
    return csv_path

class VideoExporter:
    """Coordena a geracao do MP4 analisado e de seu arquivo CSV."""

    def __init__(self, options, progress=None):
        self.options = options
        self.progress = progress

    def export(self, analysis, output_path):
        output_path = Path(output_path)
        if output_path.resolve() == analysis.input_path.resolve():
            raise ValueError('A saida nao pode sobrescrever o video original.')
        analysis.capture_originals()
        identity = SessionStore.source_identity(analysis.input_path)
        if analysis.source_identity and identity != analysis.source_identity:
            raise ValueError('O video foi alterado desde a deteccao.')
        analysis.source_identity = identity
        if analysis.session_path is None:
            analysis.session_path = output_path.with_name(f'{output_path.stem}_{analysis.session_id[:8]}.session.zip')
        staged = output_path.with_name(f'.{output_path.stem}.{uuid4().hex}.tmp.mp4')
        staged_csv = staged.with_suffix('.csv')
        try:
            _render_video(analysis, staged, self.options, self.progress, published_output=output_path)
            if not staged.is_file() or staged.stat().st_size == 0:
                raise ValueError('Video de saida vazio.')
            ArtifactPublisher().publish([(staged, output_path), (staged_csv, output_path.with_suffix('.csv'))])
            analysis.record('export', output=str(output_path), frames=len(analysis.full_precision_rows))
            SessionStore().save(analysis, options=self.options)
        finally:
            for temporary in (staged, staged_csv):
                if temporary.exists():
                    temporary.unlink()
        return output_path.with_suffix('.csv')

def render_video(data, out_path, opts, progress=None):
    """Fachada publica preservada para nao alterar o codigo da interface."""
    return VideoExporter(opts, progress).export(data, out_path)
