import pytest

from ..ratingmodel import PowerLawRating, SplineRating
from .. import data

@pytest.mark.parametrize('ratingmodel', ['powerlaw', 'spline'])
@pytest.mark.parametrize('method', ['advi', 'nuts'])
@pytest.mark.parametrize('segments, dof', [(1, 3), (4, 8)])
def test_rating_fit(ratingmodel, method, segments, dof):
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

    trace = rating.fit(method=method)
    df_model = rating.table(trace)

    assert len(df_model) > 0
    assert all(df_model.stage >= 0)
    assert all(df_model.discharge >= 0)
