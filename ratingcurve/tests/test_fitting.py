import numpy as np
import pymc as pm
import pytest

from .. import data
from .._compat import merge_idata
from ..ratings import PowerLawRating, SplineRating

# these tests assert on structure, not convergence, so they fit briefly
SHORT_FIT = 20_000


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


@pytest.fixture(params=['uniform', 'normal'])
def rating(request):
    """A two-segment power law rating under each supported breakpoint prior."""
    if request.param == 'uniform':
        return PowerLawRating(segments=2)

    return PowerLawRating(segments=2,
                          prior={'distribution': 'normal',
                                 'mu': [2.0, 8.0],
                                 'sigma': [0.5, 1.0]})


def test_save_and_load(tmp_path):
    """
    Test that a fitted rating survives a save/load round trip.

    Exercises the inference data groups, which are an `arviz.InferenceData`
    on ArviZ 0.x and an `xarray.DataTree` on ArviZ 1.x.
    """
    df = data.load('green channel')

    rating = PowerLawRating(segments=2)
    _ = rating.fit(df['stage'], df['q'], q_sigma=df['q_sigma'], n=SHORT_FIT)
    file = tmp_path / 'rating.nc'
    rating.save(file)

    loaded = PowerLawRating.load(file)

    # the posterior is stored, so it must survive the round trip exactly
    for var in rating.idata.posterior.data_vars:
        np.testing.assert_allclose(loaded.idata.posterior[var].values,
                                   rating.idata.posterior[var].values)

    # `load` rebuilds the model, so it must clear the seeds too
    with loaded.model:
        assert 'log_likelihood' in pm.compute_log_likelihood(loaded.idata,
                                                             progressbar=False)

    # the loaded model must still be usable; the table is resampled, so
    # compare its shape rather than its (stochastic) values
    table = loaded.table(h=np.array([3.0, 6.0, 9.0]))
    assert len(table) > 0
    assert all(table.stage >= 0)
    assert all(table.discharge >= 0)


def test_fitted_model_has_default_initial_values():
    """
    Test that a fitted model works with PyMC's graph transformations.

    PyMC refuses to convert a model whose variables carry explicit initial
    values, which breaks `pm.compute_log_likelihood` and, in turn, the model
    comparison in the tutorials. The breakpoint seeds are cleared once the
    model no longer needs them. Clearing does not depend on the prior, so one
    is enough here; `test_breakpoints_are_seeded` covers both.
    """
    df = data.load('green channel')

    rating = PowerLawRating(segments=2)
    _ = rating.fit(df['stage'], df['q'], q_sigma=df['q_sigma'], n=SHORT_FIT)

    initial_values = rating.model.rvs_to_initial_values
    assert all(v is None for v in initial_values.values())

    with rating.model:
        idata = pm.compute_log_likelihood(rating.idata, progressbar=False)

    assert 'log_likelihood' in idata


def test_breakpoints_are_seeded(rating):
    """
    Test that the breakpoints are seeded on the variable when the model is
    built.

    The seeds are cleared once the model is ready, so guard the build: losing
    them still fits and still produces a plausible rating, silently starting
    the sampler from the midpoint of the breakpoint bounds instead.
    """
    df = data.load('green channel')

    rating.build_model(df['stage'], df['q'], q_sigma=df['q_sigma'])

    initial_values = rating.model.rvs_to_initial_values
    seeded = {rv.name: v for rv, v in initial_values.items() if v is not None}

    assert 'hs_' in seeded, 'breakpoints were not seeded'
    np.testing.assert_allclose(seeded['hs_'], rating._init_hs)


def test_merge_idata_keeps_existing_groups():
    """
    Test that merging leaves groups that are already present alone.

    `InferenceData.extend` joins 'left' and keeps them, whereas
    `DataTree.update` lets the incoming group win, so the two ArviZ versions
    would otherwise disagree about what a re-prediction writes into idata.
    """
    with pm.Model():
        x = pm.Normal('x', 0, 1)
        pm.Normal('y', mu=x, sigma=1, observed=[1.0, 2.0])

        idata = pm.sample_prior_predictive(draws=20, random_seed=0)
        before = np.asarray(idata['prior']['x']).ravel()[0]

        # a second, different prior sample must not displace the first
        merge_idata(idata, pm.sample_prior_predictive(draws=20, random_seed=1))
        after = np.asarray(idata['prior']['x']).ravel()[0]

    assert before == after
