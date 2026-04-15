import pytest
import numpy as np

from ..ratings import PowerLawRating, SplineRating
from .. import data


def test_nuts_fit():
    """
    Test fitting a power law by NUTS.

    NUTS is slow, so test a limited functionality.
    """

    df = data.load('green channel')

    rating = PowerLawRating(method='nuts')

    _ = rating.fit(df['stage'], df['q'], q_sigma=df['q_sigma'])
    df_model = rating.table()

    assert len(df_model) > 0
    assert all(df_model.stage >= 0)
    assert all(df_model.discharge >= 0)


@pytest.mark.parametrize('ratingmodel', ['powerlaw', 'spline'])
@pytest.mark.parametrize('segments, dof', [(1, 3), (4, 8)])
def test_advi_fit(ratingmodel, segments, dof):
    """
    Test fitting a power law by ADVI.
    """
    df = data.load('green channel')

    if ratingmodel == 'powerlaw':
        rating = PowerLawRating(segments=segments)
    elif ratingmodel == 'spline':
        rating = SplineRating(df=dof)

    _ = rating.fit(df['stage'], df['q'],
                   q_sigma=df['q_sigma'], method='advi')
    df_model = rating.table()

    assert len(df_model) > 0
    assert all(df_model.stage >= 0)
    assert all(df_model.discharge >= 0)


def test_no_zero_flows():
    """
    Test that a zero flow raises an error.
    """
    q = np.array([0, 1, 2])
    h = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        rating = PowerLawRating()
        _ = rating.fit(h, q)


def test_equation():
    """Test that equation() returns denormalized parameters that reproduce
    the rating table."""
    df = data.load('green channel')
    rating = PowerLawRating(segments=2)
    rating.fit(df['stage'], df['q'], q_sigma=df['q_sigma'])

    params = rating.equation()

    assert 'a' in params
    assert 'b' in params
    assert 'hs' in params
    assert len(params['b']) == 2
    assert len(params['hs']) == 2

    # Verify equation reproduces table output
    table = rating.table()
    h = table['stage'].values

    ho = np.ones(2)
    ho[0] = 0

    log_q = params['a']
    for i in range(2):
        log_q = log_q + params['b'][i] * np.log(
            np.clip(h - params['hs'][i], 0, np.inf) + ho[i]
        )

    q_eq = np.exp(log_q)

    # The equation with posterior means should approximate the table median.
    # Use a loose tolerance since mean of posterior != exact median prediction.
    np.testing.assert_allclose(q_eq, table['median'].values, rtol=0.15)


def test_zero_flow_prior():
    """
    Test the zero-flow prior.

    The first breakpoint should be below the lowest observed flow.
    """
    df = data.load('green channel')

    q_min = df['q'].min()

    with pytest.raises(ValueError):
        rating = PowerLawRating(segments=1,
                                prior={'distribution': 'normal',
                                       'mu': [q_min],
                                       'sigma': [1]})
        _ = rating.fit(df['stage'], df['q'])
