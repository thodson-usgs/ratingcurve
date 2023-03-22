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
#from .plot import PowerLawPlotMixin, SplinePlotMixin
#from .sklearn import RegressorMixin
from .ratingmodel import PowerLawRating, RatingData

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
    
