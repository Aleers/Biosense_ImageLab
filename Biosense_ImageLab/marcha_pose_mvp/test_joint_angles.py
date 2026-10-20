import csv
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from joint_angles import JointAngleCalculator, angle_2d
from csv_exporter import CsvExporter

class AngleTests(unittest.TestCase):
    def test_geometry(self):
        self.assertAlmostEqual(angle_2d((1, 0), (0, 0), (0, 1)), 90)
        self.assertAlmostEqual(angle_2d((-1, 0), (0, 0), (1, 0)), 180)
        self.assertTrue(math.isnan(angle_2d((0, 0), (0, 0), (1, 0))))
        self.assertTrue(math.isnan(angle_2d((float('nan'), 0), (0, 0), (1, 0))))

    def test_joint_visibility_and_sides(self):
        points = np.ones((33, 3))
        for indices in ((11, 23, 25, 27, 31), (12, 24, 26, 28, 32)):
            for i, index in enumerate(indices):
                points[index, :2] = (i, 0)
        calculator = JointAngleCalculator()
        self.assertEqual(len(calculator.calculate(points)), 6)
        self.assertEqual(calculator.calculate(points, ('left',)), {
            'left_hip_deg': 180., 'left_knee_deg': 180., 'left_ankle_deg': 180.})
        points[25, 2] = 0.1
        self.assertFalse(any(key.startswith('left') for key in calculator.calculate(points)))
        points[25, 2] = 1
        points[25, 0] = float('inf')
        self.assertFalse(any(key.startswith('left') for key in calculator.calculate(points)))
        self.assertEqual(calculator.calculate(None), {})

    def test_csv_precision_and_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.csv'
            fields = ['frame', 'time_s', 'angle', 'phase', 'missing']
            CsvExporter().export(path, fields, [dict(frame=2, time_s=0.0168, angle=123.456, phase='APOIO', missing=float('nan'))])
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], fields)
            self.assertEqual(rows[1], ['2', '0.0', '123.5', 'APOIO', ''])

if __name__ == '__main__':
    unittest.main()
