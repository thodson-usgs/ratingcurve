import pytest
import numpy as np

from ..ratingmodel import PowerLawRating, SplineRating
from .. import data

def test_nuts_fit():
    """Test fitting a power law by NUTS

    NUTS is slow, so test a limited functionality"""

    df = data.load('green channel')

    rating = PowerLawRating(q=df['q'],
                            h=df['stage'],
                            q_sigma=df['q_sigma'],
                            segments=2)

    trace = rating.fit(method='nuts')
    df_model = rating.table(trace)

    assert len(df_model) > 0
    assert all(df_model.stage >= 0)
    assert all(df_model.discharge >= 0)

@pytest.mark.parametrize('ratingmodel', ['powerlaw', 'spline'])
@pytest.mark.parametrize('segments, dof', [(1, 3), (4, 8)])

def test_advi_fit(ratingmodel, segments, dof):
    """Test fitting a power law by ADVI
    """
    df = data.load('green channel')

    if ratingmodel == 'powerlaw':
        rating = PowerLawRating(q=df['q'],
                                h=df['stage'],
                                q_sigma=df['q_sigma'],
                                segments=segments)
    elif ratingmodel == 'spline':
        rating = SplineRating(q=df['q'],
                              h=df['stage'],
                              q_sigma=df['q_sigma'],
                              df=dof)

    trace = rating.fit(method='advi')
    df_model = rating.table(trace)

    assert len(df_model) > 0
    assert all(df_model.stage >= 0)
    assert all(df_model.discharge >= 0)

def test_no_zero_flows():
    """Test that a zero flow raises an error
    """
    q = np.array([0, 1, 2])
    h = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        rating = PowerLawRating(q=q,
                                h=h,
                                segments=1)


def test_zero_flow_prior():
    """Test the zero-flow prior

    The first breakpoint should be below the lowest observed flow.
    """
    df = data.load('green channel')

    q_min = df['q'].min()
    
    with pytest.raises(ValueError):
        rating = PowerLawRating(q=df['q'],
                                h=df['stage'],
                                segments=1,
                                prior = {'distribution': 'normal', 'mu': [q_min], 'sigma': [1]})

