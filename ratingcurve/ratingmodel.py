"""Streamflow rating models"""
from __future__ import annotations
from typing import TYPE_CHECKING

import math
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as at

from dataclasses import dataclass, asdict
from pymc import Model
from pandas import DataFrame


from .transform import LogZTransform, Dmatrix
from .plot import PowerLawPlotMixin, SplinePlotMixin
from .sklearn import RegressorMixin

if TYPE_CHECKING:
    from typing import Tuple
    from numpy.typing import ArrayLike
    from arviz import InferenceData

class Rating(Model, RegressorMixin):
    """Abstract base class for rating models
    """
    def __init__(self, q, h, name='', model=None):
        """Initialize rating model

        Parameters
        ----------
        q, h: array-like
            Input arrays of discharge (q) and gage height (h) observations.

        name : str
          Name that will be used as prefix for names of all random variables defined within model
        """
        super().__init__(name, model)

        if np.any(q <= 0):
            raise ValueError('Discharge must be positive. Zero values may be allowed in a future release.')


    def table(self, trace, h=None, step=0.01, extend=1.1) -> DataFrame:
        """Return stage-discharge rating table

        Parameters
        ----------
        trace : ArviZ InferenceData
            Trace from MCMC sampling
        h : array-like
            Stage values to compute rating table. If None, then use the range of observations.
        step : float
            Step size for stage values
        extend : float
            Extend range of discharge values by this factor

        Returns
        -------
        DataFrame
            Rating table with columns 'stage', 'discharge', and 'sigma'
        """
        if h is None:
            h = stage_range(self.h_obs.min(), self.h_obs.max() * extend, step=step)
            ratingdata = self.predict(trace, h)
            table = ratingdata.table()
            table = table[table['discharge'] <= self.q_obs.max() * extend]

        else:
            ratingdata = self.predict(trace, h)
            table = ratingdata.table()

        return table
    
    def residuals(self, trace: InferenceData) -> ArrayLike:
        """Compute residuals of rating model

        Parameters
        ----------
        trace : arviz.InferenceData
          Arviz ``InferenceData`` object containing posterior samples of model parameters.

        Returns
        -------
        residuals : array-like
          Log residuals of rating model
        """
        q_pred = self.predict(trace, self.h_obs).discharge

        if self.q_obs.ndim == 1:
            q_obs = self.q_obs.values.reshape((-1,1))
        else:
            q_obs = self.q_obs

        return np.array(np.log(q_obs) - np.log(q_pred))
    
    def predict(self, trace: InferenceData, h: ArrayLike) -> RatingData:
        """Predicts values of new data with a trained rating model

        This function uses PyMC's built in posterior predictive sampling, which
        is convenient but slow. Many descendants of this class implement faster
        predict methods using numpy that are specific to each model.

        Parameters
        ----------
        trace : arviz.InferenceData
          Arviz ``InferenceData`` object containing posterior samples of model parameters.
        h : array-like
          Stages at which to predict discharge.

        Returns
        -------
        RatingData
            dataclass with stage, discharge, and sigma.

        See: https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-out-of-sample-predictions.html
        """
        y = np.zeros_like(h)
        q_sigma = np.zeros_like(h)
        pm.set_data({'h':h,
                     'q_sigma':q_sigma,
                     'y':y},
                     model=self)
        
        prediction = pm.sample_posterior_predictive(trace, model=self)
        q_z = prediction['posterior_predictive']['mu']

        # return data to original state
        pm.set_data({'h':self.h_obs,
                     'q_sigma':self.q_sigma,
                     'y':self.y},
                     model=self)

 
        q = self.q_transform.untransform(q_z)
        return RatingData(stage=h, discharge=q)


class PowerLawRating(Rating, PowerLawPlotMixin):
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
        q, h: array-like
            Input arrays of discharge (q) and gage height (h) observations.
        q_sigma : array-like
            Input array of discharge uncertainty in units of discharge.
        segments : int
            Number of segments in the rating.
        prior : dict
            Prior knowledge of breakpoint locations.
        """

        super().__init__(q, h, name, model)

        self.segments = segments
        self.prior = prior
        self.q_obs = q
        self.q_transform = LogZTransform(self.q_obs)
        self.y = self.q_transform.transform(self.q_obs)

        COORDS = {"obs": np.arange(len(self.y)), "splines": np.arange(segments)}
        self.add_coords(COORDS)

        # transform observational uncertainty to log scale
        if q_sigma is None:
            self.q_sigma = np.zeros_like(q) # 0
        else:
            self.q_sigma = np.log(1 + q_sigma/q)

        self.h_obs = h
        
        # data
        h = pm.MutableData("h", self.h_obs)
        q_sigma = pm.MutableData("q_sigma", self.q_sigma)
        y = pm.MutableData("y", self.y)

        # parameters
        # taking the log of h0_offset produces the clipping boundaries in Fig 1, from Reitan et al. 2019
        self.ho = np.ones((self.segments, 1))
        self.ho[0] = 0

        # priors
        w_mu = np.zeros(self.segments)
        # see Le Coz 2014 for default values, but typically between 1.5 and 2.5
        w_mu[0] = 1.6
        w = pm.Normal("w", mu=w_mu, sigma=0.5, dims="splines")
        a = pm.Normal("a", mu=0, sigma=3)
        sigma = pm.HalfCauchy("sigma", beta=0.1)

        # set priors on break points
        if self.prior['distribution'] == 'normal':
            hs = self.set_normal_prior()
        elif self.prior['distribution'] == 'uniform':
            hs = self.set_uniform_prior()
        else:
            raise NotImplementedError('Prior distribution not implemented')

        # likelihood
        b = pm.Deterministic('b', at.log( at.clip(h - hs, 0, np.inf) + self.ho))
        mu = pm.Normal("mu", a + at.dot(w, b), sigma + q_sigma, observed=y)

    def set_normal_prior(self):
        """Normal prior for breakpoints

        Sets an expected value for each breakpoint (mu) with uncertainty (sigma).
        This can be very helpful when convergence is poor.

        prior={'distribution': 'normal', 'mu': [], 'sigma': []}
        """

        self.__set_hs_bounds()
        self._init_hs = np.sort(np.array(self.prior['mu']))
        self._init_hs = self._init_hs.reshape((self.segments, 1))

        prior_mu = np.array(self.prior['mu']).reshape((self.segments, 1))
        prior_sigma = np.array(self.prior['sigma']).reshape((self.segments, 1))

        # check that the priors are within their bounds
        h_min = self.h_obs.min()
        h_max = self.h_obs.max()

        if np.any(prior_mu[0] >= h_min):
            raise ValueError('The prior mean (mu) of the first breakpoint represents '
                             'the stage of zero-flow, so must be below the lowest '
                             'observed stage.')

        if np.any(prior_mu[1:] < h_min) or np.any(prior_mu > h_max):
            raise ValueError('The prior means (mu) of subsequent breakpoints must '
                             'be within the bounds of the observed stage.')
        
        if np.any(prior_sigma < 0):
            raise ValueError('Prior standard deviations must be positive.')
        

        hs_ = pm.TruncatedNormal('hs_',
                                 mu=prior_mu,
                                 sigma=prior_sigma,
                                 lower=self._hs_lower_bounds,
                                 upper=self._hs_upper_bounds,
                                 shape=(self.segments, 1),
                                 initval=self._init_hs)

        # Sorting reduces multimodality. The benifit increases with fewer observations.
        hs = pm.Deterministic('hs', at.sort(hs_, axis=0))
        return  hs

    def set_uniform_prior(self):
        """Uniform prior for breakpoints

        Make no prior assumption about the location of the breakpoints, only their number.

        prior={distribution:'uniform', initval: []}

        TODO: clean this up
        """
        self.__set_hs_bounds()
        self.__init_hs()
        
        hs_ = pm.Uniform('hs_',
                         lower=self._hs_lower_bounds,
                         upper=self._hs_upper_bounds,
                         shape=(self.segments, 1),
                         initval=self._init_hs)

        # Sorting reduces multimodality. The benifit increases with fewer observations.
        hs = pm.Deterministic('hs', at.sort(hs_, axis=0))
        return hs
    
    def predict(self, trace: InferenceData, h: ArrayLike) -> RatingData:
        """Predicts values of new data with a trained rating model

        This is a faster but model-specific version of the generic predict method.
        If the PowerLawRating model changes, so must this function.

        Parameters
        ----------
        trace : ArviZ InferenceData
        h : array-like
            Stages at which to predict discharge.

        Returns
        -------
        RatingData
            Dataframe with columns 'stage', 'discharge', and 'sigma' containing predicted discharge and uncertainty.
        """
        trace = az.extract(trace)
        sample = trace.sample.shape[0]
        a = trace['a'].values
        w = trace['w'].values
        w2 = np.expand_dims(w.T, -1) #FIX
        hs = np.moveaxis(trace['hs'].values, -1, 0)
        sigma = trace['sigma'].values

        h_tile = np.tile(h, sample).reshape(sample, 1, -1)

        b = np.log( np.clip(h_tile - hs, 0, np.inf) + self.ho)
        q_z = a + (b*w2).sum(axis=1).T 
        e = np.random.normal(0, sigma, sample)
        q = self.q_transform.untransform(q_z + e)

        return RatingData(stage=h, discharge=q)

    def __set_hs_bounds(self, n: int=1):
        """Set upper and lower bounds for breakpoints

        Sets the lower and upper bounds for the breakpoints. For the first
        breakpoint, the lower bound is set to 0. The upper bound is set to the
        minimum observed stage. For the remaining breakpoints, the lower bound
        is set to the minimum observed stage. The upper bound is set to the
        maximum observed stage.

        Parameters
        ----------
        n : int, optional
            Number of observations to exlude from each segment, by default 1
        """
        e = 1e-6
        h = np.sort(self.h_obs)
        self._hs_lower_bounds = np.zeros(self.segments) 
        self._hs_lower_bounds[1:] = h[n * np.arange(1, self.segments) - 1] + e
        self._hs_lower_bounds = self._hs_lower_bounds.reshape((-1, 1))

        self._hs_upper_bounds = np.zeros(self.segments)
        self._hs_upper_bounds[0] = h[0]
        self._hs_upper_bounds[:0:-1] = h[-n * np.arange(1, self.segments)]
        self._hs_upper_bounds = self._hs_upper_bounds.reshape((-1, 1))  - e

    def __init_hs(self):
        """Initialize breakpoints
        
        """
        self._init_hs = self.prior.get('initval', None)

        # TODO: distribute to evenly split the data
        if self._init_hs is None:
            self._init_hs = np.random.rand(self.segments, 1) \
                * (self._hs_upper_bounds - self._hs_lower_bounds) \
                + self._hs_lower_bounds
            self._init_hs = np.sort(self._init_hs, axis=0)  # not necessary?

        else:
            self._init_hs = np.sort(np.array(self._init_hs)).reshape((self.segments, 1))


class SplineRating(Rating, SplinePlotMixin):
    """Natural spline rating model
    """

    def __init__(self, q, h, q_sigma=None, mean=0, sd=1, df=5, name='', model=None):
        """Create a natural spline rating model

        Parameters
        ----------
        q, h: array-like
            Input arrays of discharge (q) and stage (h) observations.
        q_sigma : array-like, optional
            Input array of discharge uncertainty in units of discharge.
        knots : array-like
            Stage value locations of the spline knots.
        mean, sd : float
            Prior mean and standard deviation for the spline coefficients.
        df : int
            Degrees of freedom for the spline coefficients.
        """
        super().__init__(q, h, name, model)
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
        COORDS = {"obs": np.arange(len(self.y)), "splines": np.arange(self.B.shape[1])}
        self.add_coords(COORDS)

        # data
        B = pm.MutableData("B", self.B)
        q_sigma = pm.MutableData("q_sigma", self.q_sigma)
        y = pm.MutableData("y", self.y)

        # priors
        w = pm.Normal("w", mu=mean, sigma=sd, dims="splines")
        sigma = pm.HalfCauchy("sigma", beta=0.1) + q_sigma

        # likelihood
        mu = pm.Normal("mu", at.dot(B, w.T), sigma, observed=self.y, dims="obs")

    def predict(self, trace: InferenceData, h: ArrayLike) -> RatingData:
        """Predicts values of new data with a trained rating model

        Parameters
        ----------
        trace : ArviZ InferenceData
        h : array-likea
            Stages at which to predict discharge.

        Returns
        -------
        RatingData
            dataclass with stage, discharge, and sigma.
        """
        trace = az.extract(trace)
        sample = trace.sample.shape[0]
        w = trace['w'].values.squeeze()
        B = self.d_transform(h)
        sigma = trace['sigma'].values
        q_z = np.dot(B, w)
        e = np.random.normal(0, sigma, sample)

        q = self.q_transform.untransform(q_z + e)
        return RatingData(stage=h, discharge=q)


class RatingData:
    """Dataclass for rating model output"""
    def __init__(self, stage, discharge):
        """Input stochastic predictions
        """
        self.stage = stage.squeeze()
        self.discharge = discharge.squeeze()

    def mean(self, axis: int = 1) -> ArrayLike:
        """Return expected (mean) discharge
        """
        return np.mean(self.discharge, axis=axis)
 
    def median(self, axis: int = 1) -> ArrayLike:
        """Return median discharge
        """
        return np.median(self.discharge, axis=axis)

    def gse(self, axis: int = 1) -> ArrayLike:
        """Return geometric standard error

        References
        ----------
        .. [1] Kirkwood, T. B., "Geometric means and measures of dispersion",
               Biometrics, vol. 35, pp. 908-909, 1979
        """
        z = np.log(self.discharge)
        return np.exp(z.std(axis=axis))
    
    def prediction_interval(self, alpha: float = 0.05) -> Tuple[ArrayLike, ArrayLike]:
        """ Return prediction interval

        See Table 1 of [1]_ for the definition of the prediction interval.

        References
        ----------
        .. [1] Limpert, E, et al., "Log-normal distributions across the sciences:
               Keys and Clues", BioScience, vol. 51 (5), pp. 341-352, 2001.
        """
        median = self.median()
        gse = self.gse()
        return (median / gse**1.96,
                median * gse**1.96)
 
    def table(self) -> DataFrame:
        """Return a rating table

        Return a rating table with stage, expected discharge, median discharge,
        and gse.

        Returns
        -------
        DataFrame
        """
        return DataFrame({'stage': self.stage.round(2),
                          'discharge': self.mean().round(2),
                          'median': self.median().round(2),
                          'gse': self.gse().round(4)})


def stage_range(minimum: float, maximum: float, step: float = 0.01):
    """Returns a range of stage values

    To compute the range, round down (up) to the nearest step for
    the minumum (maximum).

    Parameters
    ----------
    h_min, h_max : float
        Minimum and maximum stage (h) observations.
    """
    start = minimum - math.remainder(minimum, step)
    stop = maximum + (step - math.remainder(maximum, step))

    return np.arange(start, stop, step)
