import numpy as np
import pytest

from ..transform import LogZTransform, Transform, UnitTransform, ZTransform, compute_knots


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


@pytest.mark.parametrize('minimum, maximum', [(0, 1e2), (0, 1e4),
                                             (10, 1e2), (10, 1e4),
                                             (1e3, 1e4)])
@pytest.mark.parametrize('n', [1, 5, 10, 50, 1000])
def test_compute_knots(minimum, maximum, n):
    knots = compute_knots(minimum, maximum, n)
    assert knots[0] == minimum
    assert knots[-1] <= maximum
    assert len(knots) == n


@pytest.mark.parametrize('minimum, maximum', [(0, 0), (10, 0), (1e3, 0),
                                              (1e3, 1e2)])
def test_compute_knots_rejects_empty_range(minimum, maximum):
    """
    Test that knots spanning nothing are refused.

    `np.linspace` runs backwards without complaint, so an unordered pair
    would otherwise yield descending or repeated knots.
    """
    with pytest.raises(ValueError):
        compute_knots(minimum, maximum, 5)


@pytest.mark.parametrize('x', [[0.0, 1.0, 2.0], [-1.0, 1.0], [0.0]])
def test_logztransform_rejects_non_positive(x):
    """
    Test that a non-positive value is refused rather than silently spread.

    The log of one is undefined, and it would carry into the mean and
    standard deviation, so every transformed value would come back nan.
    """
    with pytest.raises(ValueError):
        LogZTransform(np.array(x))
