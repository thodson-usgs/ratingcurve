import pytest

import numpy as np

from ..ratingmodel import compute_knots, stage_range

from ..ratingmodel import Rating, PowerLawRating, SplineRating

# TODO specify these with pytest parametrize
def test_compute_knots():
    assert np.allclose(compute_knots(0, 1, 2), np.array([0, 1]))
    assert np.allclose(compute_knots(0, 1, 3), np.array([0, 0.5, 1]))
    assert np.allclose(compute_knots(0, 1, 11), np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))

def test_stage_range():
    assert np.allclose(stage_range(0, 1, 2), np.array([0])) # Nonsense
    assert np.allclose(stage_range(0.01, 1.01, 0.1), np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]))
    assert np.allclose(stage_range(0.01, 1.01, 0.2), np.array([0. , 0.2, 0.4, 0.6, 0.8, 1.]))
    
