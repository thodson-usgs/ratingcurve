"""Experimental streamflow rating models using PyMC ModelBuilder."""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
import pytensor.tensor as at

from .transform import Dmatrix
from .plot import PowerLawPlotMixin, SplinePlotMixin
from .ratingmodel_builder import RatingModelBuilder

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class PowerLawRating(RatingModelBuilder, PowerLawPlotMixin):
    """
    Multi-segment power law rating using Heaviside parameterization.
    """
    # Give the model a name
    _model_type = "PowerLawRating"

    # And a version
    version = "0.1"
        
    @staticmethod
    def get_default_model_config() -> dict:
        """
        Returns a `model_config` dictionary with all the required model configuration parameters
        needed to build the model. It will be passed to the class instance on
        initialization, in case the user doesn't provide any model_config of their own.

        When specified by a user, `model_config` in the `PowerLawRatingModel` must be a
        dictionary and contain the two keys `segements` and `prior`. These two keys are formatted
        as follows:

        segments : int
            Number of segments in the rating.
        prior : dict
            Prior knowledge of breakpoint locations. Must contain the key `distribution`,
            which can either be set to a `'uniform'` or `'normal'` distribution. If a normal distribution,
            then the mean `mu` and width `sigma` must be given as well.

        Examples:
        ``model_config = {'segments': 2, 'prior': {'distribution': 'uniform'}}``
        or
        ``model_config = {'segments': 2, 'prior': {'distribution': 'normal', 'mu': [1, 2], 'sigma':[1, 1]}}``
        or
        ``model_config = {'segments': 4, 'prior': {'distribution': 'normal', 'mu': [1, 2, 5, 9], 'sigma':[1, 1, 1, 1]}}``

        Note that the number of normal distribution means and widths must be the same as
        the number of segments. Additionally, the first mean must be less than the lowest
        observed stage as it defines the stage of zero flow.
        """
        model_config = {'segments': 2, 'prior': {'distribution': 'uniform'}}

        return model_config

    
    def build_model(self, h: ArrayLike, q: ArrayLike, q_sigma: ArrayLike=None, **kwargs):
        """
        Creates the PyMC model.

        Parameters
        ----------
        h : array_like
            Input array of gage height (h) observations.
        q : array_like
            Input array of discharge (q) observations.
        q_sigma : array_like, optional
            Input array of discharge uncertainty in units of discharge.
        """
        # Pre-process data: Converts q to normalized log space
        # (and q_sigma to log space if given) 
        self.segments = self.model_config.get('segments')
        self._generate_and_preprocess_model_data(h, q, q_sigma)

        # Create the model
        with pm.Model(coords=self.model_coords) as self.model:
            h = pm.MutableData("h", self.h_obs)
            logq = pm.MutableData("logq", self.log_q_z)
            q_sigma = pm.MutableData("q_sigma", self.q_sigma)

            # parameters
            # taking the log of h0_offset produces the clipping boundaries in Fig 1, from Reitan et al. 2019
            self.ho = np.ones((self.segments, 1))
            self.ho[0] = 0
    
            # priors
            b_mu = np.zeros(self.segments)
            # see Le Coz 2014 for default values, but typically between 1.5 and 2.5
            b_mu[0] = 1.6
            b = pm.Normal("b", mu=b_mu, sigma=0.5, dims="splines")
            a = pm.Normal("a", mu=0, sigma=3)
            sigma = pm.HalfCauchy("sigma", beta=0.1)
     
            # Set priors on break points
            if self.model_config.get('prior').get('distribution', 'uniform') == 'normal':
                hs = self.set_normal_prior()
            elif self.model_config.get('prior').get('distribution', 'uniform') == 'uniform':
                hs = self.set_uniform_prior()
            else:
                raise NotImplementedError('Prior distribution not implemented')

            # likelihood
            X = pm.Deterministic('X', at.log( at.clip(h - hs, 0, np.inf) + self.ho))
            obs = pm.Normal("model_q", a + at.dot(b, X), sigma + q_sigma, shape=h.shape, observed=logq)

    
    def set_normal_prior(self):
        """
        Normal prior for breakpoints. Sets an expected value for each
        breakpoint (mu) with uncertainty (sigma). This can be very helpful
        when convergence is poor.

        prior={'distribution': 'normal', 'mu': [], 'sigma': []}
        """
        self.__set_hs_bounds()
        self._init_hs = np.sort(np.array(self.model_config.get('prior').get('mu')))
        self._init_hs = self._init_hs.reshape((self.segments, 1))

        prior_mu = np.array(self.model_config.get('prior').get('mu')).reshape((self.segments, 1))
        prior_sigma = np.array(self.model_config.get('prior').get('sigma')).reshape((self.segments, 1))

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
        """
        Uniform prior for breakpoints. Make no prior assumption about 
        the location of the breakpoints, only the number of breaks and
        that the breakpoints are ordered.

        prior={distribution:'uniform', initval: []}
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
    

    def __set_hs_bounds(self, n: int=1):
        """
        Set upper and lower bounds for breakpoints

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
        """
        Initialize breakpoints by randomly selecting points within the stage data
        range. Selected points are then sorted.
        """
        self._init_hs = self.model_config.get('prior').get('initval', None)

        # TODO: distribute to evenly split the data
        if self._init_hs is None:
            self._init_hs = np.random.rand(self.segments, 1) \
                * (self._hs_upper_bounds - self._hs_lower_bounds) \
                + self._hs_lower_bounds
            self._init_hs = np.sort(self._init_hs, axis=0)  # not necessary?

        else:
            self._init_hs = np.sort(np.array(self._init_hs)).reshape((self.segments, 1))



class SplineRating(RatingModelBuilder, SplinePlotMixin):
    """
    Natural spline rating.
    """
    # Give the model a name
    _model_type = "SplineRating"

    # And a version
    version = "0.1"
        
    @staticmethod
    def get_default_model_config() -> dict:
        """
        Returns a `model_config` dictionary with all the required model configuration parameters
        needed to build the model. It will be passed to the class instance on
        initialization, in case the user doesn't provide any model_config of their own.

        When specified by a user, `model_config` in the `SplineRatingModel`  must be a
        dictionary and contain the three keys `mean`, `sd`, and `df`. These three keys are
        formatted as follows:

        mean : float
            Mean of the normal prior for the spline coefficients.
        sd : float
            Standard deviation of the normal prior for the spline coefficients.
        df : int
            Degrees of freedom for the spline coefficients.

        Examples:
        ``model_config = {'mean': 0, 'ds': 1, 'df': 5}``
        """
        model_config = {'mean': 0, 'ds': 1, 'df': 5}

        return model_config

    
    def _data_setter(self, h: ArrayLike, q: ArrayLike=None, q_sigma: ArrayLike=None):
        """
        Update _data_setter to include spline design matrix update.

        Parameters
        ----------
        h : array_like
            Input training array of gage height (h) observations.
        q : array_like, optional
            Target discharge (q) values.
        q_sigma : array_like, optional
            Discharge uncertainty in units of discharge.
        """
        super()._data_setter(h, q, q_sigma)

        with self.model:
            pm.set_data({'B': self.d_transform(np.array(h))})


    
    def build_model(self, h: ArrayLike, q: ArrayLike, q_sigma: ArrayLike=None, **kwargs):
        """
        Creates the PyMC model.

        Parameters
        ----------
        h : array_like
            Input array of gage height (h) observations.
        q : array_like
            Input array of discharge (q) observations.
        q_sigma : array_like, optional
            Input array of discharge uncertainty in units of discharge.
        """
        # Need to compute the design matrix now as we need it to get the number
        # of "segments" to include when preprocessing data for coords.
        self.h_obs = np.array(h).flatten()
        self._dmatrix = Dmatrix(self.h_obs, self.model_config.get('df'), 'cr')
        self.d_transform = self._dmatrix.transform
        self.B = self.d_transform(self.h_obs)
        self.segments = self.B.shape[1]

        # Pre-process data: Converts q to normalized log space
        # (and q_sigma to log space if given) 
        self._generate_and_preprocess_model_data(h, q, q_sigma)

        # Create the model
        with pm.Model(coords=self.model_coords) as self.model:
            h = pm.MutableData("h", self.h_obs)
            logq = pm.MutableData("logq", self.log_q_z)
            q_sigma = pm.MutableData("q_sigma", self.q_sigma)
            B = pm.MutableData("B", self.B)
    
            # priors
            w = pm.Normal("w", mu=self.model_config.get('mean'), sigma=self.model_config.get('sd'), dims="splines")
            sigma = pm.HalfCauchy("sigma", beta=0.1)
    
            # likelihood
            obs = pm.Normal("model_q", at.dot(B, w.T), sigma + q_sigma, shape=h.shape, observed=logq)
            