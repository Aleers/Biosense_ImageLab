import math

import numpy as np

from processor import angle


def test_right_angle():
    assert math.isclose(angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1])), 90.0)


def test_straight_angle():
    assert math.isclose(angle(np.array([-1, 0]), np.array([0, 0]), np.array([1, 0])), 180.0)
