import pytest
import numpy as np

from ..transform import *

@pytest.mark.parametrize('length', [2, 5, 10, 100, 1000, 10_000])
@pytest.mark.parametrize('range', [0.1, 1, 5, 10, 1e2, 1e4])
def test_transform(length, range):
    x = np.random.rand(length) * range
    t = Transform(x)
    assert np.allclose(t.transform(x), x)
    assert np.allclose(t.untransform(t.transform(x)), x)


@pytest.mark.parametrize('length', [2, 5, 10, 100, 1000, 10_000])
@pytest.mark.parametrize('range', [0.1, 1, 5, 10, 1e2, 1e4])
def test_ztransform(length, range):
    x = np.random.rand(length) * range
    zt = ZTransform(x)
    assert np.allclose(zt.transform(x), (x - np.mean(x))/ np.std(x))
    assert np.allclose(zt.untransform(zt.transform(x)), x)


@pytest.mark.parametrize('length', [2, 5, 10, 100, 1000, 10_000])
@pytest.mark.parametrize('range', [0.1, 1, 5, 10, 1e2, 1e4])
def test_logztransform(length, range):
    x = np.random.rand(length) * range

    if (x == 0).any():
        pytest.xfail("Drew an exact zero which is not implemented")

    lzt = LogZTransform(x)
    assert np.allclose(lzt.transform(x), (np.log(x) - np.mean(np.log(x)))/ np.std(np.log(x)))
    assert np.allclose(lzt.untransform(lzt.transform(x)), x)


@pytest.mark.parametrize('length', [2, 5, 10, 100, 1000, 10_000])
@pytest.mark.parametrize('range', [0.1, 1, 5, 10, 1e2, 1e4])
def test_unittransform(length, range):
    x = np.random.rand(length) * range
    ut = UnitTransform(x)
    assert np.allclose(ut.transform(x), x / np.max(x))
    assert np.allclose(ut.untransform(ut.transform(x)), x)


@pytest.mark.parametrize('minimum', [0, 10, 1e3])
@pytest.mark.parametrize('maximum', [0, 1e2, 1e4])
@pytest.mark.parametrize('n', [1, 5, 10, 50, 1000])
def test_compute_knots(minimum, maximum, n):
    if minimum >= maximum:
        pytest.xfail("minimum must be less than maximum")
 
    knots = compute_knots(minimum, maximum, n)
    assert knots[0] == minimum
    assert knots[-1] <= maximum
    assert len(knots) == n