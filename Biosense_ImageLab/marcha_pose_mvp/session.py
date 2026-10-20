"""Versioned recoverable sessions, original observations and audit history."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import io
import json
import os
import tempfile
import sys
import zipfile
import numpy as np
SCHEMA_VERSION = 1

def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value

class SessionStore:
    @staticmethod
    def source_identity(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        return dict(sha256=digest.hexdigest(), size=Path(path).stat().st_size)

    @staticmethod
    def software_identity():
        root = Path(__file__).parent
        versions = {}
        for name in ('numpy', 'mediapipe', 'opencv-contrib-python', 'Pillow'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = 'unavailable'
        import cv2
        return dict(python=sys.version, opencv_runtime=cv2.__version__, dependencies=versions, files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.glob('*.py')) if not p.name.startswith('test_')})

    def save(self, data, path=None, options=None):
        path = Path(path or data.session_path)
        if not data.landmarks or not np.isfinite(data.fps) or data.fps <= 0:
            raise ValueError('Sessao sem quadros ou FPS valido.')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not data.raw_landmarks:
            data.capture_originals()
        if not data.source_identity:
            data.source_identity = self.source_identity(data.input_path)
        if not data.software_identity:
            data.software_identity = self.software_identity()
        arrays = {}
        for prefix, sequence in [('raw', data.raw_landmarks), ('current', data.landmarks)]:
            arrays[prefix+'_present'] = np.array([p is not None for p in sequence], dtype=bool)
            arrays[prefix] = np.stack([np.zeros((33, 3), np.float32) if p is None else p for p in sequence])
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        manifest = json_safe(dict(schema_version=SCHEMA_VERSION, session_id=data.session_id,
            saved_at=datetime.now(timezone.utc).isoformat(), input_path=str(data.input_path.resolve()),
            source_identity=data.source_identity, software_identity=data.software_identity,
            processing_identity=self.software_identity(),
            fps=data.fps, width=data.width, height=data.height, timestamps=data.timestamps,
            timestamp_source='nominal_fps', predominant_side=data.predominant_side,
            calibration_points=data.calibration_points, calibration_cm=data.calibration_cm,
            manual_segments=data.manual_segments, tracked_segments=data.tracked_segments,
            tracking_records=data.tracking_records, review_history=data.review_history, audit=data.audit,
            gait_records=data.gait_records, display_records=data.display_records,
            options=asdict(options) if options else data.saved_options, metrics=data.full_precision_rows,
            units=dict(angles='internal_2d_degrees', knee_foot_offset='cm' if data.px_per_cm else 'px'),
            landmark_payload_sha256=hashlib.sha256(stream.getvalue()).hexdigest()))
        descriptor, temp_name = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
        os.close(descriptor)
        try:
            with zipfile.ZipFile(temp_name, 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, allow_nan=False))
                archive.writestr('landmarks.npz', stream.getvalue())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        data.session_path = path
        if options:
            data.saved_options = asdict(options)
        return path

    def load(self, path, input_path=None, verify_source=True):
        from processor import Analysis
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            payload = archive.read('landmarks.npz')
        if manifest['schema_version'] != SCHEMA_VERSION:
            raise ValueError('Versao de sessao nao suportada.')
        if hashlib.sha256(payload).hexdigest() != manifest['landmark_payload_sha256']:
            raise ValueError('Dados da sessao corrompidos.')
        source = Path(input_path or manifest['input_path'])
        if verify_source and self.source_identity(source) != manifest['source_identity']:
            raise ValueError('O video nao corresponde ao original da sessao.')
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            sequences = {}
            for prefix in ('raw', 'current'):
                values, present = arrays[prefix], arrays[prefix+'_present']
                if values.ndim != 3 or values.shape[1:] != (33, 3) or len(values) != len(present):
                    raise ValueError('Landmarks invalidos na sessao.')
                sequences[prefix] = [p.copy() if exists else None for p, exists in zip(values, present)]
        if len(sequences['raw']) != len(sequences['current']):
            raise ValueError('Sequencias inconsistentes na sessao.')
        data = Analysis(source, manifest['fps'], manifest['width'], manifest['height'], sequences['current'])
        data.capture_originals(sequences['raw'])
        for name in ('session_id', 'source_identity', 'software_identity', 'timestamps', 'predominant_side',
                     'calibration_cm', 'manual_segments', 'tracked_segments', 'tracking_records', 'audit'):
            setattr(data, name, manifest[name])
        points = manifest['calibration_points']
        data.calibration_points = tuple(tuple(p) for p in points) if points else None
        data.review_history = manifest.get('review_history', [])
        data.gait_records = manifest.get('gait_records', [])
        data.display_records = manifest.get('display_records', [])
        if len(manifest['timestamps']) != len(data.landmarks) or not np.all(np.isfinite(manifest['timestamps'])):
            raise ValueError('Timestamps invalidos na sessao.')
        data.saved_options = manifest['options'] or {}
        data.full_precision_rows = manifest['metrics']
        data.session_path = Path(path)
        return data
