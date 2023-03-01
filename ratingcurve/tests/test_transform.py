import pytest
import numpy as np

from ..transform import *

class TestTransform:
    def test_transform(self):
        x = np.array([1, 2, 3, 4, 5])
        t = Transform(x)
        assert np.allclose(t.transform(x), x)
        assert np.allclose(t.untransform(t.transform(x)), x)


class TestZTransform:
    def test_transform(self):
        x = np.array([1, 2, 3, 4, 5])
        z = ZTransform(x)
        assert np.allclose(z.transform(x), np.array([-1.41421356, -0.70710678, 0., 0.70710678, 1.41421356]))
        assert np.allclose(z.untransform(z.transform(x)), x)


class TestLogZTransform:
    def test_transform(self):
        x = np.array([1, 2, 3, 4, 5])
        lz = LogZTransform(x)
        assert np.allclose(lz.transform(x), np.array([-1.68450008, -0.46506562,  0.24825781,  0.75436884,  1.14693905]))
        assert np.allclose(lz.untransform(lz.transform(x)), x)


class TestUnitTransform:
    def test_transform(self):
        x = np.array([1, 2, 3, 4, 5])
        u = UnitTransform(x)
        assert np.allclose(u.transform(x), np.array([0.2, 0.4, 0.6, 0.8, 1.]))
        assert np.allclose(u.untransform(u.transform(x)), x)


def test_compute_knots():
    assert np.allclose(compute_knots(0, 1, 2), np.array([0, 1]))
    assert np.allclose(compute_knots(0, 1, 3), np.array([0, 0.5, 1]))
    assert np.allclose(compute_knots(0, 1, 11), np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))