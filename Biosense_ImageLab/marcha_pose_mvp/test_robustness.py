"""Regression scenarios for data quality, recovery and tracking failure modes."""
import io
import json
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from joint_angles import JointAngleCalculator
from landmark_quality import LandmarkValidator
from processor import Analysis, ProcessingOptions, PoseVideoAnalyzer, ear_center, gait_phases, render_video, stabilize_lower_limbs, stepdown_metrics
from session import SessionStore
from temporal_filter import TemporalLandmarkFilter
from tracking import LandmarkTracker, TrackingOptions


def body():
    points = np.ones((33, 3), dtype=np.float32)
    points[:, :2] = np.arange(66).reshape(33, 2)+50
    for indices, x in [((11, 23, 25, 27, 29, 31), 80), ((12, 24, 26, 28, 30, 32), 180)]:
        for offset, index in enumerate(indices):
            points[index, :2] = (x+offset*2, 40+offset*20)
    return points


def marker_frame(positions=()):
    frame = np.zeros((160, 240, 3), np.uint8)
    for point in positions:
        cv2.circle(frame, tuple(map(int, point)), 5, (255, 255, 255), -1)
    return frame


class QualityTests(unittest.TestCase):
    def test_shared_exclusion(self):
        points = body()
        points[27, 2] = 0.1
        self.assertNotIn('left_knee_deg', JointAngleCalculator().calculate(points))
        self.assertNotIn('left_fppa_deg', stepdown_metrics(points))
        self.assertIsNone(LandmarkValidator().distance(points, (27, 28), 2))
        points[7, :2] = np.nan
        points[8, 2] = 0
        self.assertIsNone(ear_center(points))

    def test_invalid_shape_indices_and_scale(self):
        validator = LandmarkValidator()
        self.assertEqual(validator.assess(None, (0,)).reasons, ('person_missing',))
        self.assertFalse(validator.valid(np.zeros((3, 2)), (0,)))
        self.assertFalse(validator.valid(body(), (40,)))
        for cm in (0, -1, np.nan):
            self.assertIsNone(validator.scale(((0, 0), (10, 0)), cm))
        self.assertIsNone(validator.scale(((0, 0), (0, 0)), 10))
        self.assertEqual(validator.scale(((0, 0), (10, 0)), 5), 2)

    def test_short_and_occluded_gait(self):
        for size in (0, 1, 3, 7):
            data = Analysis(Path('video'), 30, 240, 160, [body() for _ in range(size)])
            if size > 1:
                data.landmarks[1] = None
            phases = gait_phases(data)
            for side in ('left', 'right'):
                self.assertEqual(len(phases[side]['stance']), size)
                if size > 1:
                    self.assertFalse(phases[side]['valid'][1])
                    self.assertEqual(phases[side]['event'][1], '')
                    if size > 2:
                        self.assertEqual(phases[side]['event'][2], '')

    def test_side_swap_and_originals(self):
        initial = body()
        switched = initial.copy()
        for a, b in [(11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
            switched[[a, b]] = switched[[b, a]]
        data = Analysis(Path('video'), 30, 240, 160, [initial.copy(), switched.copy()])
        data.capture_originals()
        stabilize_lower_limbs(data)
        np.testing.assert_array_equal(data.landmarks[1], initial)
        np.testing.assert_array_equal(data.raw_landmarks[1], switched)
        self.assertFalse(data.raw_landmarks[1].flags.writeable)
        self.assertEqual(data.audit[0]['action'], 'side_identity_correction')

    def test_no_identity_guess_across_gap(self):
        initial = body()
        switched = initial.copy()
        switched[[23, 24]] = switched[[24, 23]]
        data = Analysis(Path('video'), 30, 240, 160, [initial, None, switched.copy()])
        stabilize_lower_limbs(data)
        np.testing.assert_array_equal(data.landmarks[2], switched)


class TrackingTests(unittest.TestCase):
    def run_frames(self, frames, sequence=None, options=None):
        sequence = sequence or [body() for _ in frames]
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, frame) for frame in frames]
        tracker = MagicMock()
        tracker.update.return_value = (False, None)
        with patch('tracking.cv2.VideoCapture', return_value=cap), patch('tracking.cv2.TrackerCSRT_create', return_value=tracker, create=True):
            batch = LandmarkTracker(options).run('video', sequence, 0, 27, np.array([80., 80.]), 30)
        cap.release.assert_called_once()
        return batch

    def test_continuous_motion_without_mutation(self):
        frames = [marker_frame([(80+3*i, 80)]) for i in range(5)]
        sequence = [body() for _ in frames]
        before = [p.copy() for p in sequence]
        batch = self.run_frames(frames, sequence)
        self.assertEqual([record['state'] for record in batch.records], ['manual']+['tracking']*4)
        for index, point in batch.updates:
            np.testing.assert_allclose(point[:2], [80+3*index, 80], atol=0.2)
        for original, current in zip(before, sequence):
            np.testing.assert_array_equal(original, current)

    def test_occlusion_and_confirmed_reacquisition(self):
        frames = [marker_frame([(80, 80)]), marker_frame(), marker_frame([(80, 80)]), marker_frame([(80, 80)])]
        batch = self.run_frames(frames)
        self.assertEqual([r['state'] for r in batch.records], ['manual', 'lost', 'suspect', 'reacquired'])
        self.assertEqual(batch.updates[1][1][2], 0)
        self.assertEqual(batch.updates[2][1][2], 0)
        self.assertGreater(batch.updates[3][1][2], 0.45)

    def test_competing_targets_not_arbitrarily_selected(self):
        tracker = LandmarkTracker()
        template = tracker.patch(cv2.cvtColor(marker_frame([(80, 80)]), cv2.COLOR_BGR2GRAY), (80, 80))
        candidate, reason, _ = tracker.select_marker(marker_frame([(65, 80), (95, 80)]), (80, 80), template)
        self.assertIsNone(candidate)
        self.assertEqual(reason, 'competing_targets')
        batch = self.run_frames([marker_frame([(80, 80)]), marker_frame([(65, 80), (95, 80)])])
        self.assertEqual(batch.records[1]['state'], 'suspect')

    def test_csrt_drift_is_rejected(self):
        cap, tracker = MagicMock(), MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, marker_frame([(80, 80)])), (True, marker_frame())]
        tracker.update.return_value = (True, (62, 62, 36, 36))
        with patch('tracking.cv2.VideoCapture', return_value=cap), patch('tracking.cv2.TrackerCSRT_create', return_value=tracker, create=True):
            result = LandmarkTracker().run('video', [body(), body()], 0, 27, (80, 80), 30)
        self.assertEqual(result.records[1]['state'], 'lost')

    def test_missing_pose_still_advances_tracker(self):
        frames = [marker_frame([(80, 80)]), marker_frame([(83, 80)]), marker_frame([(86, 80)])]
        batch = self.run_frames(frames, [body(), None, body()])
        self.assertEqual([index for index, _ in batch.updates], [0, 2])
        np.testing.assert_allclose(batch.updates[-1][1][:2], [86, 80], atol=0.2)

    def test_failed_read_releases_and_does_not_mutate(self):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, marker_frame([(80, 80)])), (False, None)]
        points = body()
        before = points.copy()
        with patch('tracking.cv2.VideoCapture', return_value=cap):
            with self.assertRaisesRegex(ValueError, 'Falha de leitura'):
                LandmarkTracker().run('video', [points, body()], 0, 27, (80, 80), 30)
        cap.release.assert_called_once()
        np.testing.assert_array_equal(points, before)

    def test_cancel_does_not_commit(self):
        stop = threading.Event()
        stop.set()
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, marker_frame([(80, 80)]))
        with patch('tracking.cv2.VideoCapture', return_value=cap):
            batch = LandmarkTracker().run('video', [body(), body()], 0, 27, (80, 80), 30, stop)
        self.assertTrue(batch.cancelled)
        cap.release.assert_called_once()

    def test_optical_flow_recovers_marker_below_white_threshold(self):
        initial = marker_frame([(80, 80)])
        dimmed = marker_frame([(83, 80)])
        dimmed[dimmed > 0] = 180
        baseline = self.run_frames([initial, dimmed])
        enhanced = self.run_frames([initial, dimmed], options=TrackingOptions(use_optical_flow=True))
        self.assertEqual(baseline.records[1]['state'], 'lost')
        self.assertEqual(enhanced.records[1]['source'], 'optical_flow')
        np.testing.assert_allclose(enhanced.updates[-1][1][:2], [83, 80], atol=0.5)

    def test_optical_flow_translation(self):
        tracker = LandmarkTracker(TrackingOptions(use_optical_flow=True))
        first = cv2.cvtColor(marker_frame([(80, 80)]), cv2.COLOR_BGR2GRAY)
        second = cv2.cvtColor(marker_frame([(83, 82)]), cv2.COLOR_BGR2GRAY)
        predicted = tracker.optical_prediction(first, second, (80, 80))
        self.assertIsNotNone(predicted)
        np.testing.assert_allclose(predicted, [83, 82], atol=0.3)


class FilterTests(unittest.TestCase):
    def test_reduces_stationary_noise(self):
        random = np.random.default_rng(5)
        noise = random.normal(0, 2, 90)
        smoother = TemporalLandmarkFilter()
        values = []
        for i, error in enumerate(noise):
            points = body()
            points[27, 0] = 100+error
            values.append(smoother.update(points, i/30)[27, 0]-100)
        self.assertLess(np.std(values[10:]), np.std(noise[10:]))

    def test_no_interpolation_across_missing_or_low_quality(self):
        smoother = TemporalLandmarkFilter()
        initial = body()
        smoother.update(initial, 0)
        self.assertIsNone(smoother.update(None, 1/30))
        moved = initial.copy()
        moved[:, 0] += 100
        np.testing.assert_array_equal(smoother.update(moved, 2/30), moved)
        moved[27, 2] = 0
        low = smoother.update(moved, 3/30)
        self.assertEqual(low[27, 2], 0)
        valid = initial.copy()
        np.testing.assert_array_equal(smoother.update(valid, 4/30)[27], valid[27])

    def test_same_time_constant_across_frame_rates(self):
        def response(fps):
            smoother = TemporalLandmarkFilter()
            points = body()
            points[27, 0] = 0
            smoother.update(points, 0)
            points[27, 0] = 10
            for i in range(1, fps//5+1):
                result = smoother.update(points, i/fps)
            return result[27, 0]
        self.assertAlmostEqual(response(30), response(60), places=4)


class SessionTests(unittest.TestCase):
    def test_round_trip_precision_originals_and_source_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source.mp4'
            source.write_bytes(b'original source')
            data = Analysis(source, 30, 240, 160, [body(), None])
            data.capture_originals()
            original = data.raw_landmarks[0].copy()
            data.landmarks[0][27, 0] = 123.456
            data.record('tracking_result', landmark=27, after=[(0, data.landmarks[0][27])])
            data.full_precision_rows = [dict(frame=0, time_s=1/30, angle=123.456)]
            data.tracking_records = [dict(frame=0, landmark=27, state='manual')]
            data.review_history = [dict(type='point', landmark=27, snapshot=[(0, original[27])], previous_records=[])]
            path = Path(folder)/'sample.session.zip'
            SessionStore().save(data, path, ProcessingOptions())
            loaded = SessionStore().load(path)
            np.testing.assert_array_equal(loaded.raw_landmarks[0], original)
            np.testing.assert_array_equal(loaded.landmarks[0], data.landmarks[0])
            self.assertIsNone(loaded.landmarks[1])
            self.assertFalse(loaded.raw_landmarks[0].flags.writeable)
            self.assertEqual(loaded.audit, data.audit)
            self.assertEqual(loaded.review_history[0]['landmark'], 27)
            np.testing.assert_array_equal(loaded.review_history[0]['snapshot'][0][1], original[27])
            self.assertEqual(loaded.full_precision_rows[0]['angle'], 123.456)
            self.assertEqual(loaded.timestamps, [0, 1/30])
            source.write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'nao corresponde'):
                SessionStore().load(path)

    def test_atomic_save_preserves_previous_session(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source.mp4'
            source.write_bytes(b'source')
            data = Analysis(source, 30, 240, 160, [body()])
            path = Path(folder)/'sample.session.zip'
            store = SessionStore()
            store.save(data, path)
            previous = path.read_bytes()
            data.record('review_action', kind='manual')
            with patch('session.os.replace', side_effect=OSError('simulated failure')):
                with self.assertRaises(OSError):
                    store.save(data, path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertFalse(list(Path(folder).glob('*.tmp')))

    def test_payload_corruption_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source.mp4'
            source.write_bytes(b'source')
            path = Path(folder)/'sample.session.zip'
            SessionStore().save(Analysis(source,30,240,160,[body()]),path)
            with zipfile.ZipFile(path) as archive:
                manifest = archive.read('manifest.json')
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('manifest.json',manifest)
                archive.writestr('landmarks.npz',b'corrupt')
            with self.assertRaisesRegex(ValueError,'corrompidos'):
                SessionStore().load(path)


class VideoTests(unittest.TestCase):
    def test_real_codec_export_and_legacy_csv(self):
        import csv
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
            if not writer.isOpened():
                self.skipTest('mp4v unavailable')
            for _ in range(3):
                writer.write(np.zeros((480,640,3),np.uint8))
            writer.release()
            data = Analysis(source,30,640,480,[body(),None,body()])
            data.calibration_points=((0,0),(100,0))
            data.calibration_cm=10
            out = Path(folder)/'result.mp4'
            csv_path=render_video(data,out,ProcessingOptions(side='Ambos'))
            with csv_path.open(newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),3)
            self.assertEqual(rows[1]['left_knee_deg'],'')
            self.assertEqual(rows[1]['left_phase'],'')
            self.assertIn('left_knee_foot_offset_cm',rows[0])
            self.assertTrue(all(len(v.split('.')[1])<=1 for row in rows for v in row.values() if '.' in v))
            restored=SessionStore().load(data.session_path)
            self.assertEqual(len(restored.full_precision_rows),3)
            cap=cv2.VideoCapture(str(out))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),3)
            cap.release()

    def test_failed_export_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source.mp4'
            source.write_bytes(b'source')
            out = Path(folder)/'result.mp4'
            out.write_bytes(b'previous video')
            out.with_suffix('.csv').write_bytes(b'previous csv')
            data = Analysis(source,30,640,480,[body(), body()])
            cap=MagicMock()
            cap.isOpened.return_value=True
            cap.read.side_effect=[(True,np.zeros((480,640,3),np.uint8)),(False,None)]
            writer=MagicMock()
            writer.isOpened.return_value=True
            with patch('processor.cv2.VideoCapture',return_value=cap), patch('processor.cv2.VideoWriter',return_value=writer):
                with self.assertRaisesRegex(ValueError,'Exportacao incompleta'):
                    render_video(data,out,ProcessingOptions())
            self.assertEqual(out.read_bytes(),b'previous video')
            self.assertEqual(out.with_suffix('.csv').read_bytes(),b'previous csv')
            cap.release.assert_called_once()
            writer.release.assert_called_once()

    def test_detector_read_failure_releases(self):
        cap=MagicMock()
        cap.isOpened.return_value=True
        cap.get.side_effect=lambda prop: {cv2.CAP_PROP_FPS:30,cv2.CAP_PROP_FRAME_WIDTH:240,cv2.CAP_PROP_FRAME_HEIGHT:160,cv2.CAP_PROP_FRAME_COUNT:2}.get(prop,0)
        cap.read.side_effect=[(True,marker_frame()),(False,None)]
        model=MagicMock()
        model.process.return_value.pose_landmarks=None
        context=MagicMock()
        context.__enter__.return_value=model
        with patch('processor.cv2.VideoCapture',return_value=cap), patch('processor.mp.solutions.pose.Pose',return_value=context):
            with self.assertRaisesRegex(ValueError,'Leitura incompleta'):
                PoseVideoAnalyzer().analyze(Path('video'))
        cap.release.assert_called_once()

class ReviewCommitTests(unittest.TestCase):
    def test_commit_and_undo_preserve_originals_and_quality(self):
        from types import SimpleNamespace, MethodType
        from reviewer import ReviewWindow
        from tracking import TrackingBatch
        data=Analysis(Path('video'),30,240,160,[body(),body()])
        data.capture_originals()
        before=[(i,p[27].copy()) for i,p in enumerate(data.landmarks)]
        original=data.raw_landmarks[0].copy()
        fake=SimpleNamespace(data=data,opts=ProcessingOptions(view='Frontal'),index=0,
            undo_stack=data.review_history,undo_button=MagicMock(),measure_button=MagicMock(),
            landmark_measure_button=MagicMock(),info=MagicMock(),show=MagicMock(),persist=lambda:True,tracking=False)
        fake.push_action=MethodType(ReviewWindow.push_action,fake)
        value=before[0][1].copy()
        value[:2]=[100,100]
        batch=TrackingBatch(updates=[(0,value)],records=[dict(frame=0,landmark=27,state='manual')])
        ReviewWindow.tracking_done(fake,27,before,batch)
        np.testing.assert_array_equal(data.landmarks[0][27,:2],[100,100])
        self.assertEqual(len(data.review_history),1)
        ReviewWindow.undo(fake)
        np.testing.assert_array_equal(data.landmarks[0][27],before[0][1])
        np.testing.assert_array_equal(data.raw_landmarks[0],original)
        self.assertEqual(data.tracking_records,[])
        self.assertEqual(data.audit[-1]['action'],'undo')

    def test_failed_save_keeps_review_open(self):
        from types import SimpleNamespace
        from reviewer import ReviewWindow
        from app import MarchaApp
        fake=SimpleNamespace(persist=lambda:False,closed=MagicMock(),destroy=MagicMock(),tracking=False)
        ReviewWindow.close(fake)
        fake.closed.set.assert_not_called()
        fake.destroy.assert_not_called()
        MarchaApp._export(SimpleNamespace(),fake)
        fake.destroy.assert_not_called()

class PublicationTests(unittest.TestCase):
    def test_failure_rolls_back_related_files(self):
        import os
        from artifact_io import ArtifactPublisher
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            targets=[root/'final.mp4',root/'final.csv']
            sources=[root/'staged.mp4',root/'staged.csv']
            for target in targets:
                target.write_bytes(b'old')
            for source in sources:
                source.write_bytes(b'new')
            real_replace=os.replace
            def failing_replace(source,target):
                if Path(source)==sources[1]:
                    raise OSError('publication failure')
                return real_replace(source,target)
            with patch('artifact_io.os.replace',side_effect=failing_replace):
                with self.assertRaises(OSError):
                    ArtifactPublisher().publish(list(zip(sources,targets)))
            for target in targets:
                self.assertEqual(target.read_bytes(),b'old')
            self.assertFalse(list(root.glob('*.backup')))

if __name__ == '__main__':
    unittest.main()
