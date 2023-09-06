"""Experimental streamflow rating models"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as at

from dataclasses import dataclass, asdict
from pymc import Model
from pandas import DataFrame


from .transform import LogZTransform, Dmatrix
from .plot import PowerLawPlotMixin, SplinePlotMixin
#from .sklearn import RegressorMixin
from .ratingmodel import Rating, PowerLawRating, RatingData

if TYPE_CHECKING:
    from arviz import InferenceData
    from numpy.typing import ArrayLike


class ReitanRating(PowerLawRating):
    """Experimental multi-segment power law rating using the Reitan parameterization.

    Unlike Reitan Eq. 5, this version uses a fixed offset for each segment (ho).
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

        super(PowerLawRating, self).__init__(q, h, name, model)

        self.segments = segments
        self.prior = prior
        self.q_obs = q
        self.q_transform = LogZTransform(self.q_obs)
        self.y = self.q_transform.transform(self.q_obs)

        COORDS = {"obs": np.arange(len(self.y)), "splines": np.arange(segments)}
        self.add_coords(COORDS)

        # transform observational uncertainty to log scale
        if q_sigma is None:
            self.q_sigma = 0
        else:
            self.q_sigma = np.log(1 + q_sigma/q)

        self.h_obs = h

        # observations
        h = pm.MutableData("h", self.h_obs)
        q_sigma = pm.MutableData("q_sigma", self.q_sigma)

        # fixed parameters
        # taking the log of h0_offset produces the clipping boundaries in Fig 1, from Reitan et al. 2019
        self._ho = np.ones((self.segments, 1))
        self._ho[0] = 0

        # priors
        # see Le Coz 2014 for default values, but typically between 1.5 and 2.5
        #w = pm.Normal("w", mu=1.6, sigma=0.5, dims="splines")
        w = pm.TruncatedNormal("w", mu=1.6, sigma=1.0, lower=0.1, dims="splines") # lower is somewhat arbitrary
        a = pm.Normal("a", mu=0, sigma=2)

        # set priors on break points
        if self.prior['distribution'] == 'normal':
            hs = self.set_normal_prior()
        elif self.prior['distribution'] == 'uniform':
            hs = self.set_uniform_prior()
        else:
            raise NotImplementedError('Prior distribution not implemented')

        # likelihood
        ho = self._ho
        inf = at.constant([np.inf], dtype='float64').reshape((-1, 1 ))
        hs1 = at.concatenate([hs, inf])

        b = at.log( at.clip(h - hs, 0, hs1[1:] - hs) + ho) #best but suspect ho is accumulating (ho added to each segment)
        sigma = pm.HalfCauchy("sigma", beta=0.1)
        mu = pm.Normal("mu", a + at.dot(w, b), sigma + q_sigma, observed=self.y)

    def predict(self, trace: InferenceData, h: ArrayLike) -> RatingData:
        """Predicts values of new data with a trained rating model

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

        clips = np.zeros((hs.shape[1], 1))
        clips[0] = -np.inf
        h_tile = np.tile(h, sample).reshape(sample, 1, -1)

        ho = self._ho
        inf = np.array(np.inf).reshape(-1, 1)
        hs1 = np.pad(hs, ((0,0), (0,1), (0,0)), 'constant', constant_values=np.inf )
        b = np.log( np.clip(h - hs, 0, hs1[:,1:] - hs) + ho)
        q_z = a + (b*w2).sum(axis=1).T
        e = np.random.normal(0, sigma, sample)

        return self._format_ratingdata(h=h, q_z=q_z+e)


class LeCozRating(PowerLawRating):
    """Experimental multi-segment power law rating using the LeCoz (2014) parameterization.

    """
    def __init__(
        self,
        q,
        h,
        segments,
        prior={'distribution': 'uniform'},
        q_sigma=None,
        m=None,
        name='',
        model=None):
        """Create a multi-segment power law rating model

        Parameters
        ----------
        q, h: array-like
            Input arrays of discharge (q) and gage height (h) observations.
        q_sigma : array-like
            Input array of discharge uncertainty in units of discharge.
        m : array-like
            Hydrologic control matrix.
        segments : int
            Number of segments in the rating.
        prior : dict
            Prior knowledge of breakpoint locations.
        """

        super(PowerLawRating, self).__init__(q, h, name, model)

        self.segments = segments
        self.prior = prior
        self.q_obs = q
        self.q_transform = LogZTransform(self.q_obs)
        self.y = self.q_transform.transform(self.q_obs)

        COORDS = {"obs": np.arange(len(self.y)), "splines": np.arange(segments)}
        self.add_coords(COORDS)

        # transform observational uncertainty to log scale
        if q_sigma is None:
            self.q_sigma = 0
        else:
            self.q_sigma = np.log(1 + q_sigma/q)

        if m is None:
            self.m = np.eye(self.segments)
        else:
            self.m = at.constant(m)

        self.h_obs = h
        # observations
        h = pm.MutableData("h", self.h_obs)
        q_sigma = pm.MutableData("q_sigma", self.q_sigma)

        # fixed parameters
        # taking the log of h0_offset produces the clipping boundaries in Fig 1, from Reitan et al. 2019
        #self._ho = np.ones((self.segments, 1))
        #self._ho[0] = 0

        # priors
        # see Le Coz 2014 for default values, but typically between 1.5 and 2.5
        #w = pm.Normal("w", mu=1.6, sigma=0.5, dims="splines")
        w = pm.TruncatedNormal("w", mu=1.6, sigma=1.0, lower=0.1, dims="splines") # lower is somewhat arbitrary
        a = pm.Normal("a", mu=0, sigma=3, dims="splines")

        # set priors on break points
        if self.prior['distribution'] == 'normal':
            hs = self.set_normal_prior()
        elif self.prior['distribution'] == 'uniform':
            hs = self.set_uniform_prior()
        else:
            raise NotImplementedError('Prior distribution not implemented')

        # likelihood
        inf = at.constant([np.inf], dtype='float64').reshape((-1, 1 ))
        hs1 = at.concatenate([hs, inf])

        #i = at.switch( (h > hs) & (h <= hs1[1:]), 1, 0)
        #x = at.clip( h - hs, 1e-6, hs1[1:] - hs )
        #b = a + w * at.log(x).T
        ##q = at.sum(i * at.dot(self.m, at.exp(b.T)), axis=0)
        #q = at.sum(i * at.exp(b.T), axis=0)
        i = at.switch( (h > hs) & (h <= hs1[1:]), 1, 0)
        x = at.clip( h - hs, 1e-6, hs1[1:] - hs )
        b = at.exp(a) * (x).T**w
        q = at.sum(i * at.dot(self.m, b.T), axis=0)

        sigma = pm.HalfCauchy("sigma", beta=0.1)
        mu = pm.Normal("mu", at.log(q), sigma + q_sigma, observed=self.y)

    def predict(self, trace: InferenceData, h: ArrayLike) -> RatingData:
        """Predicts values of new data with a trained rating model

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

        clips = np.zeros((hs.shape[1], 1))
        clips[0] = -np.inf
        h_tile = np.tile(h, sample).reshape(sample, 1, -1)

        ho = self._ho
        inf = np.array(np.inf).reshape(-1, 1)
        hs1 = np.pad(hs, ((0,0), (0,1), (0,0)), 'constant', constant_values=np.inf )
        b = np.log( np.clip(h - hs, 0, hs1[:,1:] - hs) + ho)
        q_z = a + (b*w2).sum(axis=1).T
        e = np.random.normal(0, sigma, sample)

        return self._format_ratingdata(h=h, q_z=q_z+e)



class ISORating(PowerLawRating):
    """Experimental multi-segment power law rating using the Reitan parameterization.

    Unlike Reitan Eq. 5, this version uses a fixed offset for each segment (ho).
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

        super(PowerLawRating, self).__init__(q, h, name, model)

        self.segments = segments
        self.prior = prior
        self.q_obs = q
        self.q_transform = LogZTransform(self.q_obs)
        self.y = self.q_transform.transform(self.q_obs)

        COORDS = {"obs": np.arange(len(self.y)), "splines": np.arange(segments)}
        self.add_coords(COORDS)

        # transform observational uncertainty to log scale
        if q_sigma is None:
            self.q_sigma = 0
        else:
            self.q_sigma = np.log(1 + q_sigma/q)

        self.h_obs = h

        # observations
        h = pm.MutableData("h", self.h_obs)
        q_sigma = pm.MutableData("q_sigma", self.q_sigma)

        # priors
        # see Le Coz 2014 for default values, but typically between 1.5 and 2.5
        w = pm.TruncatedNormal("w", mu=2, sigma=0.4, lower=0.5, dims="splines") # lower is somewhat arbitrary
        a = pm.Normal("a", mu=0, sigma=3, dims='splines') # a is scale dependent
        #TEST a = pm.Normal("a", mu=0, sigma=3) # a is scale dependent

        # set priors on break points
        if self.prior['distribution'] == 'normal':
            hs = self.set_normal_prior()
        elif self.prior['distribution'] == 'uniform':
            hs = self.set_uniform_prior()
        else:
            raise NotImplementedError('Prior distribution not implemented')

        # likelihood
        inf = at.constant([np.inf], dtype='float64').reshape((-1, 1 ))
        hs1 = at.concatenate([hs, inf])
        x = at.clip(h - hs, 1e-6, hs1[1:] - hs) # could use at.switch instead
        #TEST b = at.exp(w * at.log(x.T))
        #TEST q = a + at.log(at.sum(b, axis=1))
        b = at.exp(a + w * at.log(x.T))
        q = at.log(at.sum(b, axis=1))

        sigma = pm.HalfCauchy("sigma", beta=0.1)
        mu = pm.Normal("mu", q, sigma + q_sigma, observed=self.y)

    def predict(self, trace: InferenceData, h: ArrayLike) -> RatingData:
        """Predicts values of new data with a trained rating model

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

        clips = np.zeros((hs.shape[1], 1))
        clips[0] = -np.inf
        h_tile = np.tile(h, sample).reshape(sample, 1, -1)

        inf = np.array(np.inf).reshape(-1, 1)
        hs1 = np.pad(hs, ((0,0), (0,1), (0,0)), 'constant', constant_values=np.inf )

        x = np.clip(h - hs, 1e-6, hs1[:,1:] - hs)
        #b = at.switch(h > hs, at.exp(a + w * at.log(x.T)), 0 )
        b = np.exp(a + w * np.log(x.T))
        q_z = np.log(np.sum(b, axis=1))
        e = np.random.normal(0, sigma, sample)

        return self._format_ratingdata(h=h, q_z=q_z+e)


class Test():
    def __init__(x=None):
        print('test')



class BrokenPowerLawRating(Rating, PowerLawPlotMixin):
    """
    Experimental multi-segment power law rating using the standard parameterization
    (see https://en.wikipedia.org/wiki/Power_law#Broken_power_law).

    This parameterization does not require the ho offset, as the positive slope
    requirement is placed into the prior.
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
        q : array-like
            Input array of discharge (q) observations.
        h : array-like
            Input array of gage height (h) observations.
        q_sigma : array-like
            Input array of discharge uncertainty in units of discharge.
        segments : int
            Number of segments in the rating. (I.e. the number of breakpoints
            minus one.)
        prior : dict
            Prior knowledge of breakpoint locations.
        """

        super().__init__(q, h, name, model)

        self.segments = segments
        self.prior = prior
        self.q_obs = q
        # self.q_transform = LogZTransform(self.q_obs)
        # self.y = self.q_transform.transform(self.q_obs)
        self.y = np.log(self.q_obs)

        COORDS = {"obs": np.arange(len(self.y)), "splines": np.arange(segments)}
        self.add_coords(COORDS)

        # transform observational uncertainty to log scale
        if q_sigma is None:
            self.q_sigma = 0
        else:
            self.q_sigma = np.log(1 + q_sigma/q)

        self.h_obs = np.array(h)

        # observations
        h = pm.MutableData("h", self.h_obs)
        q_sigma = pm.MutableData("q_sigma", self.q_sigma)
        y = pm.MutableData("y", self.y)

        # priors
        # w is the same as alpha, the power law slopes
        # lower bound of truncated normal forces discharge to increase with stage
        w = pm.Uniform("w", lower=0, upper=100, shape=(self.segments, 1), dims="splines")
        # a is the scale parameter
        # a = pm.Flat("a")
        a = pm.Uniform("a", lower=-100, upper=100)
        sigma = pm.HalfCauchy("sigma", beta=0.1)

        # set priors on break points
        if self.prior['distribution'] == 'normal':
            hs = self.set_normal_prior()
        elif self.prior['distribution'] == 'uniform':
            hs = self.set_uniform_prior()
        else:
            raise NotImplementedError('Prior distribution not implemented')

        # -1 gives w_{i-1) - w_i rather than w_i - w_{i-1} of diff
        w_diff = -1 * at.diff(w, axis=0)
        sums = at.cumsum(w_diff * at.log(hs), axis=0)
        # Sum for first element is 0, as it does not have a summation 
        sums = at.concatenate([pm.math.constant(0, ndim=2), sums])

        # Create ranges for each segment
        segments_range = at.concatenate([pm.math.constant(0, ndim=2),
                                         hs,
                                         pm.math.constant(np.inf, ndim=2)])

        # Tensors are broadcasts for vectorized computation.
        #   Calculates function within range sets value to 0 everywhere else. 
        #   Then sum along segment dimension to collapse.
        q = at.switch((h > segments_range[:-1]) & (h <= segments_range[1:]), 
                       a + w * at.log(h) + sums, 0)
        q = at.sum(q, axis=0)

        mu = pm.Normal("mu", q, sigma + q_sigma, observed=y)

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
        a = trace['a'].values.reshape((-1, 1, 1))
        w = trace['w'].values
        w_diff = np.moveaxis(w[:-1, ...] - w[1:, ...], -1, 0)
        hs = np.moveaxis(trace['hs'].values, -1, 0)
        sigma = trace['sigma'].values
        
        h_tile = np.tile(h, sample).reshape(sample, 1, -1)

        sums = np.cumsum(w_diff * np.log(hs), axis=1)
        sums = np.insert(sums, 0, 0, axis=1)

        # Create ranges for each segment
        segments_range = np.insert(np.insert(hs, 0, 0, axis=1), self.segments, np.inf, axis=1)

        # Arrays are broadcasts for vectorized computation.
        #   Calculates function within range sets value to 0 everywhere else. 
        #   Then sum along segment dimension to collapse.
        q_z = np.where((h_tile > segments_range[:, :-1]) & (h_tile <= segments_range[:, 1:]), 
                       a + np.moveaxis(w, -1, 0) * np.log(h_tile) + sums, 0)
        q_z = np.sum(q_z, axis=1).T

        e = np.random.normal(0, sigma, sample)
        # q = self.q_transform.untransform(q_z + e)
        q = np.exp(q_z + e)
        
        return RatingData(stage=h, discharge=q)

    def set_normal_prior(self):
        """
        Normal prior for breakpoints. Sets an expected value for each
        breakpoint (mu) with uncertainty (sigma). This can be very helpful
        when convergence is poor. Expected breakpoint values (mu) must be
        within the data range.

        prior={'distribution': 'normal', 'mu': [], 'sigma': []}
        """

        self._init_hs = np.sort(np.array(self.prior['mu']))
        self._init_hs = self._init_hs.reshape((self.segments - 1, 1))

        prior_mu = np.array(self.prior['mu']).reshape((self.segments - 1, 1))
        prior_sigma = np.array(self.prior['sigma']).reshape((self.segments - 1, 1))

        # check that the priors are within their bounds
        h_min = self.h_obs.min()
        h_max = self.h_obs.max()

        if np.any(prior_mu < h_min) or np.any(prior_mu > h_max):
            raise ValueError('The prior means (mu) of the breakpoints must '
                             'be within the bounds of the observed stage.')
        
        if np.any(prior_sigma < 0):
            raise ValueError('Prior standard deviations must be positive.')
        
        # Built in PyMC ordered transform forces multiple priors to be in ascending
        #   order. This means that the breakpoints will be sorted internally to PyMC.
        hs_ = pm.TruncatedNormal('hs_',
                                 mu=prior_mu,
                                 sigma=prior_sigma,
                                 lower=h_min,
                                 upper=h_max,
                                 shape=(self.segments - 1, 1),
                                 initval=self._init_hs)

        # Sorting reduces multimodality. The benifit increases with fewer observations.
        hs = pm.Deterministic('hs', at.sort(hs_, axis=0))
        return hs

    def set_uniform_prior(self):
        """
        Uniform prior for breakpoints. Make no prior assumption about 
        the location of the breakpoints, only the number of breaks and
        that the breakpoints are ordered.

        prior={distribution:'uniform', initval: []}
        """
        self.__init_hs()
        h_min = self.h_obs.min()
        h_max = self.h_obs.max()

        # Built in PyMC ordered transform forces multiple priors to be in ascending
        #   order. This means that the breakpoints will be sorted internally to PyMC.
        hs_ = pm.Uniform('hs_',
                         lower=h_min,
                         upper=h_max,
                         shape=(self.segments - 1, 1),
                         initval=self._init_hs)

        hs = pm.Deterministic('hs', at.sort(hs_, axis=0))
        return hs

    def __init_hs(self):
        """
        Initialize breakpoints by randomly selecting points within the stage data
        range. Selected points are then sorted.
        """
        self._init_hs = self.prior.get('initval', None)
        h_min = self.h_obs.min()
        h_max = self.h_obs.max()

        # TODO: distribute to evenly split the data
        if self._init_hs is None:
            self._init_hs = np.random.rand(self.segments - 1, 1) \
                            * (h_max - h_min) + h_min
            self._init_hs = np.sort(self._init_hs, axis=0)
        else:
            self._init_hs = np.sort(np.array(self._init_hs)).reshape((self.segments - 1, 1))


class SmoothPowerLawRating(BrokenPowerLawRating):
    """
    Experimental smooth multi-segment power law rating using the standard parameterization
    (see https://en.wikipedia.org/wiki/Power_law#Broken_power_law).

    This parameterization does not require the ho offset, as the positive slope
    requirement is placed into the prior.
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
        q : array-like
            Input array of discharge (q) observations.
        h : array-like
            Input array of gage height (h) observations.
        q_sigma : array-like
            Input array of discharge uncertainty in units of discharge.
        segments : int
            Number of segments in the rating. (I.e. the number of breakpoints
            minus one.)
        prior : dict
            Prior knowledge of breakpoint locations.
        """

        super(BrokenPowerLawRating, self).__init__(q, h, name, model)

        self.segments = segments
        self.prior = prior
        self.q_obs = q
        # self.q_transform = LogZTransform(self.q_obs)
        # self.y = self.q_transform.transform(self.q_obs)
        self.y = np.log(self.q_obs)

        COORDS = {"obs": np.arange(len(self.y)), "splines": np.arange(segments)}
        self.add_coords(COORDS)

        # transform observational uncertainty to log scale
        if q_sigma is None:
            self.q_sigma = 0
        else:
            self.q_sigma = np.log(1 + q_sigma/q)

        self.h_obs = np.array(h)

        # observations
        h = pm.MutableData("h", self.h_obs)
        q_sigma = pm.MutableData("q_sigma", self.q_sigma)
        y = pm.MutableData("y", self.y)

        # priors
        # see Le Coz 2014 for default values, but typically between 1.5 and 2.5
        # w is the same as alpha, the power law slopes
        # lower bound of truncated normal forces discharge to increase with stage
        w = pm.Uniform("w", lower=0, upper=100, shape=(self.segments, 1), dims="splines")
        # a is the scale parameter
        a = pm.Flat("a")
        # delta is the smoothness parameter, limit lower bound (m) to prevent floating point errors
        delta = pm.Pareto('delta', alpha=0.5, m=0.01)
        sigma = pm.HalfCauchy("sigma", beta=0.1)

        # set priors on break points
        if self.prior['distribution'] == 'normal':
            hs = self.set_normal_prior()
        elif self.prior['distribution'] == 'uniform':
            hs = self.set_uniform_prior()
        else:
            raise NotImplementedError('Prior distribution not implemented')

        w_diff = at.diff(w, axis=0)
        sum_array = (w_diff * delta) * at.log(1 + (h/hs) ** (1/delta))
        sums = at.sum(sum_array, axis=0)
        mu = pm.Normal("mu", a + at.log(h) * w[0, ...] + sums, sigma + q_sigma, observed=y)

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
        a = trace['a'].values.reshape((-1, 1))
        delta = trace['delta'].values.reshape((-1, 1, 1))
        w = trace['w'].values
        w_diff = np.moveaxis(w[1:, ...] - w[:-1, ...], -1, 0)
        hs = np.moveaxis(trace['hs'].values, -1, 0)
        sigma = trace['sigma'].values
        
        h_tile = np.tile(h, sample).reshape(sample, 1, -1)
        
        sum_array = (w_diff * delta) * np.log(1 + (h_tile/hs) ** (1/delta))
        sums = np.sum(sum_array, axis=1)
        q_z = (a + np.squeeze(np.log(h_tile)) * (w[0, 0, :]).reshape((-1, 1)) + sums).T
        e = np.random.normal(0, sigma, sample)
        q = np.exp(q_z + e)
        breakpoint()
        
        return RatingData(stage=h, discharge=q)
