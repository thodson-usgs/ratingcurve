import pytest
import numpy as np

from ..ratingmodel import stage_range

@pytest.mark.parametrize('minimum', [0, 10, 1e3])
@pytest.mark.parametrize('maximum', [0, 1e2, 1e4])
@pytest.mark.parametrize('step', [0.01, 1, 50, 1e3])
def test_stage_range(minimum, maximum, step):
    
    if minimum >= maximum:
        pytest.xfail("minimum must be less than maximum")
    if step > (maximum - minimum):
         pytest.xfail("step must be smaller than minimum/maximum difference")

    print(maximum, step)
    stg_rng = stage_range(minimum, maximum, step)
    assert stg_rng[0] <= minimum
    assert (stg_rng[-1] >= maximum) or (np.allclose(stg_rng[-1], maximum))
    assert np.allclose(stg_rng[1] - stg_rng[0], step)
    assert len(stg_rng) <= (maximum - minimum)/step + 2