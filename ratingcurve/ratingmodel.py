"""Streamflow rating models"""

import math
import numpy as np
from pandas import DataFrame, Series

import pymc as pm
from pymc import Model
import pytensor.tensor as at

from arviz import InferenceData

from .transform import LogZTransform, Dmatrix
from .plot import plot_power_law_rating, plot_spline_rating


class Rating(Model):
    def __init__(self, name='', model=None):
        super().__init__(name, model)

    def fit(self, method="advi", n=150_000):
        mean_field = pm.fit(method=method, n=n, model=self.model)
        return mean_field

    def sample(self, n_samples, n_tune):
        with self.model:
            trace = pm.sample(50_000)

    def table(self, trace, h=None, step=0.01) -> DataFrame:
        """Return stage-discharge rating table
        """
        if h is None:
            extend = 1.1
            h = stage_range(self.h_obs.min(), self.h_obs.max() * extend, step=step)

        table = self.predict(trace, h)

        return table.round({'discharge': 2, 'stage': 2, 'sigma': 4})
    
    def rating_plot(self, trace: InferenceData, ax=None):
        """TODO Decorator controlling style of rating plots
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(5, 5))

        ax.set_ylabel('Stage')
        ax.set_xlabel('Discharge')

    def predict(self) -> DataFrame:
        raise NotImplementedError

    def save(self, filename: str) -> None:
        raise NotImplementedError

    @staticmethod
    def load(filename: str) -> Model:
        """Load a saved model
        """
        raise NotImplementedError


class PowerLawRating(Rating):
    """Multi-segment power law rating using Heaviside parameterization.
    """
    def __init__(
        self,
        q,
        h,
        segments,
        prior={'distribution': 'uniform'},
        q_sigma=None,
        name='',
        model=None):
        """Create a multi-segment power law rating model

        Parameters
        ----------
        q, h: array_like
            Input arrays of discharge (q) and gage height (h) observations.
        q_sigma : array_like
            Input array of discharge uncertainty in units of discharge.
        segments : int
            Number of segments in the rating.
        prior : dict
            Prior knowledge of breakpoint locations.
        """

        super().__init__(name, model)

        self.segments = segments
        self.prior = prior
        self.q_obs = q
        self.q_transform = LogZTransform(self.q_obs)
        self.y = self.q_transform.transform(self.q_obs)

        # transform observational uncertainty to log scale
        if q_sigma is None:
            self.q_sigma = 0
        else:
            self.q_sigma = np.log(1 + q_sigma/q)

        self.h_obs = h

        self._inf = [np.inf]

        # clipping boundary
        clips = np.zeros((self.segments, 1))
        clips[0] = -np.inf
        self._clips = at.constant(clips)

        # create h0 offsets
        self._h0_offsets = np.ones((self.segments, 1))
        self._h0_offsets[0] = 0

        self.COORDS = {"obs": np.arange(len(self.y)), "splines": np.arange(segments)}

        # compute initval
        self._hs_lower_bounds = np.zeros((self.segments, 1)) + self.h_obs.min()
        self._hs_lower_bounds[0] = 0

        self._hs_upper_bounds = np.zeros((self.segments, 1)) + self.h_obs.max()
        self._hs_upper_bounds[0] = self.h_obs.min() - 1e-6  # XXX If possible, don't hard code

        # set random init on unit interval then scale based on bounds
        self._init_hs = np.random.rand(self.segments, 1) \
            * (self._hs_upper_bounds - self._hs_lower_bounds) \
            + self._hs_lower_bounds

        self._init_hs = np.sort(self._init_hs)  # not necessary?

        self.compile_model()

    def plot(self, trace, ax=None):
        plot_power_law_rating(self, trace, ax=ax)

    def set_normal_prior(self):
        """Normal prior for breakpoints

        Sets an expected value for each breakpoint (mu) with uncertainty (sigma).
        prior={type='normal', mu=[], sigma=[]}
        """
        with Model(coords=self.COORDS) as model:
            hs_ = pm.TruncatedNormal('hs_',
                                     mu=self.prior['mu'],
                                     sigma=self.prior['sigma'],
                                     lower=self._hs_lower_bounds,
                                     upper=self._hs_upper_bounds,
                                     shape=(self.segments, 1),
                                     initval=self._init_hs)

            hs = pm.Deterministic('hs', at.sort(hs_))

        return hs

    def set_uniform_prior(self):
        """Uniform prior for breakpoints

        Make no prior assumption about the location of the breakpoints, only their number.

        prior={distribution:'uniform'}
        """
        with Model(coords=self.COORDS) as model:
            hs_ = pm.Uniform('hs_',
                             lower=self._hs_lower_bounds,
                             upper=self._hs_upper_bounds,
                             shape=(self.segments, 1),
                             initval=self._init_hs)

            hs = pm.Deterministic('hs', at.sort(hs_))

        return hs

    def compile_model(self):
        with Model(coords=self.COORDS) as model:
            h = pm.MutableData("h", self.h_obs)
            w = pm.Normal("w", mu=0, sigma=3, dims="splines")
            a = pm.Normal("a", mu=0, sigma=5)

            # set prior on break points
            if self.prior['distribution'] == 'normal':
                hs = self.set_normal_prior()
            else:
                hs = self.set_uniform_prior()

            h0 = hs - self._h0_offsets
            b = pm.Deterministic('b', at.switch(at.le(h, hs), self._clips, at.log(h-h0)))

            sigma = pm.HalfCauchy("sigma", beta=1) + self.q_sigma
            mu = pm.Normal("mu", a + at.dot(w, b), sigma, observed=self.y)

    def predict(self, trace: InferenceData, h):
        """TODO verify sigma computation
        """
        chain = trace.posterior['chain'].shape[0]
        draw = trace.posterior['draw'].shape[0]

        a = trace.posterior['a'].values.reshape(chain, draw, 1)
        w = trace.posterior['w'].values.reshape(chain, draw, -1, 1)
        hs = trace.posterior['hs'].values

        clips = np.zeros((hs.shape[2], 1))
        clips[0] = -np.inf
        h_tile = np.tile(h, draw).reshape(chain, draw, 1, -1)

        h0_offset = np.ones((hs.shape[2], 1))
        h0_offset[0] = 0
        h0 = hs - h0_offset
        b1 = np.where(h_tile <= hs, clips, np.log(h_tile-h0))
        q_z = a + (b1*w).sum(axis=2)

        sigma = q_z.std(axis=1)
        q = self.q_transform.untransform(q_z)

        return DataFrame({'discharge': q.mean(axis=1).flatten(),
                          'stage': h,
                          'sigma': np.exp(sigma).flatten()})


class SplineRating(Rating):
    """Natural spline rating model
    """

    def __init__(self, q, h, q_sigma=None, mean=0, sd=1, df=5, name='', model=None):
        """Create a natural spline rating model

        Parameters
        ----------
        q, h: array_like
            Input arrays of discharge (q) and stage (h) observations.
        q_sigma : array_like
            Input array of discharge uncertainty in units of discharge.
        knots : arrak_like
            Stage value locations of the spline knots.
        """
        super().__init__(name, model)
        self.q_obs = q
        self.h_obs = h
        self.q_transform = LogZTransform(self.q_obs)
        self.y = self.q_transform.transform(self.q_obs)

        # transform observational uncertainty to log scale
        if q_sigma is None:
            self.q_sigma = 0
        else:
            self.q_sigma = np.log(1 + q_sigma/q)

        self._dmatrix = Dmatrix(self.h_obs, df, 'cr')
        self.d_transform = self._dmatrix.transform

        self.B = self.d_transform(h)
        B = pm.MutableData("B", self.B)

        COORDS = {"obs": np.arange(len(self.y)), "splines": np.arange(self.B.shape[1])}
        self.add_coords(COORDS)

        w = pm.Normal("w", mu=mean, sigma=sd, dims="splines")

        sigma = pm.HalfCauchy("sigma", beta=1) + self.q_sigma
        mu = pm.Normal("mu", at.dot(B, w.T), sigma, observed=self.y, dims="obs")

    def predict(self, trace: InferenceData, h):
        """Predict discharge given stage


        """
        w = trace.posterior['w'].values.squeeze()
        B = self.d_transform(h)
        q_z = np.dot(B, w.T)
        q = self.q_transform.untransform(q_z)
        sigma = q_z.std(axis=1)
        
        return DataFrame({'discharge': Series(q.mean(axis=1)),
                          'stage': h,
                          'sigma': Series(np.exp(sigma))})

    def plot(self, trace, ax=None):
        plot_spline_rating(self, trace, ax=ax)


def stage_range(minimum: float, maximum: float, step: float = 0.01):
    """Returns a range of stage values

    To compute the range, round down (up) to the nearest step for 
    the minumum (maximum). 

    Parameters
    ----------
    h_min, h_max : float
        Minimum and maximum stage (h) observations.
    """
    start = minimum - (minimum % step)
    stop = maximum + (maximum % step)

    return np.arange(start, stop, step)


def compute_knots(minimum: float, maximum: float, n: int):
    """Return list of spline knots

    Parameters
    ----------
    minimum, maximum : float

    n : int
        Number of knots
    """
    return np.linspace(minimum, maximum, n)
