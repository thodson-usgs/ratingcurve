"""Experimental streamflow rating models using PyMC ModelBuilder."""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
import pytensor.tensor as at

from .modelbuilder_ratings import PowerLawRating

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class ReitanRating(PowerLawRating):
    """Experimental segmented power law rating with Reitan parameterization.

    Unlike Reitan Eq. 5, this version uses a fixed offset for each
    segment (ho).
    """
    # Give the model a name
    _model_type = "ReitanRating"

    # And a version
    version = "0.1"

    def build_model(self,
                    h: ArrayLike,
                    q: ArrayLike,
                    q_sigma: ArrayLike = None,
                    **kwargs):
        """Creates the PyMC model.

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

            # observations
            h = pm.MutableData("h", self.h_obs)
            log_q_z = pm.MutableData("log_q_z", self.log_q_z)
            q_sigma = pm.MutableData("q_sigma", self.q_sigma)

            # fixed parameters
            # taking the log of h0_offset produces the clipping boundaries
            # in Fig 1, from Reitan et al. 2019
            self._ho = np.ones((self.segments, 1))
            self._ho[0] = 0

            # priors
            # see Le Coz 2014 for default values, but typically between
            # 1.5 and 2.5
            # lower is somewhat arbitrary
            w = pm.TruncatedNormal("w", mu=1.6, sigma=1.0,
                                   lower=0.1, dims="splines")
            # w = pm.Normal("w", mu=1.6, sigma=0.5, dims="splines")
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
            inf = at.constant([np.inf], dtype='float64').reshape((-1, 1))
            hs1 = at.concatenate([hs, inf])

            # best but suspect ho is accumulating (ho added to each segment)
            b = at.log(at.clip(h - hs, 0, hs1[1:] - hs) + ho)
            sigma = pm.HalfCauchy("sigma", beta=0.1)
            obs = pm.Normal('model_q', a + at.dot(w, b), np.sqrt(sigma**2 + q_sigma**2),
                            shape=h.shape, observed=log_q_z)


class LeCozRating(PowerLawRating):
    """Experimental segmented power law rating with LeCoz+2014 parameterization.
    """
    # Give the model a name
    _model_type = "LeCozRating"

    # And a version
    version = "0.1"

    def build_model(self,
                    h: ArrayLike,
                    q: ArrayLike,
                    q_sigma: ArrayLike = None,
                    **kwargs):
        """Creates the PyMC model.

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

        if 'm' in kwargs:
            self.m = at.constant(kwargs['m'])
        else:
            self.m = np.eye(self.segments)

        # Create the model
        with pm.Model(coords=self.model_coords) as self.model:

            # observations
            h = pm.MutableData("h", self.h_obs)
            log_q_z = pm.MutableData("log_q_z", self.log_q_z)
            q_sigma = pm.MutableData("q_sigma", self.q_sigma)

            # fixed parameters
            # taking the log of h0_offset produces the clipping boundaries
            #   in Fig 1, from Reitan et al. 2019
            # self._ho = np.ones((self.segments, 1))
            # self._ho[0] = 0

            # priors
            # see Le Coz 2014 for default values, but typically between
            #   1.5 and 2.5
            # lower is somewhat arbitrary
            w = pm.TruncatedNormal("w", mu=1.6, sigma=1.0,
                                   lower=0.1, dims="splines")
            # w = pm.Normal("w", mu=1.6, sigma=0.5, dims="splines")
            a = pm.Normal("a", mu=0, sigma=3, dims="splines")

            # set priors on break points
            if self.prior['distribution'] == 'normal':
                hs = self.set_normal_prior()
            elif self.prior['distribution'] == 'uniform':
                hs = self.set_uniform_prior()
            else:
                raise NotImplementedError('Prior distribution not implemented')

            # likelihood
            inf = at.constant([np.inf], dtype='float64').reshape((-1, 1))
            hs1 = at.concatenate([hs, inf])

            # i = at.switch( (h > hs) & (h <= hs1[1:]), 1, 0)
            # x = at.clip( h - hs, 1e-6, hs1[1:] - hs )
            # b = a + w * at.log(x).T
            # q = at.sum(i * at.dot(self.m, at.exp(b.T)), axis=0)
            # q = at.sum(i * at.exp(b.T), axis=0)
            i = at.switch((h > hs) & (h <= hs1[1:]), 1, 0)
            x = at.clip(h - hs, 1e-6, hs1[1:] - hs)
            b = at.exp(a) * (x).T**w
            q = at.sum(i * at.dot(self.m, b.T), axis=0)

            sigma = pm.HalfCauchy("sigma", beta=0.1)
            obs = pm.Normal('model_q', at.log(q), np.sqrt(sigma**2 + q_sigma**2),
                            shape=h.shape, observed=log_q_z)


class ISORating(PowerLawRating):
    """Experimental segmented power law rating with ISO parameterization.
    """
    # Give the model a name
    _model_type = "ISORating"

    # And a version
    version = "0.1"

    def build_model(self,
                    h: ArrayLike,
                    q: ArrayLike,
                    q_sigma: ArrayLike = None,
                    **kwargs):
        """Creates the PyMC model.

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

            # observations
            h = pm.MutableData("h", self.h_obs)
            log_q_z = pm.MutableData("log_q_z", self.log_q_z)
            q_sigma = pm.MutableData("q_sigma", self.q_sigma)

            # priors
            # see Le Coz 2014 for default values, but typically between
            #   1.5 and 2.5
            # lower is somewhat arbitrary
            w = pm.TruncatedNormal("w", mu=2, sigma=0.4,
                                   lower=0.5, dims="splines")
            # a is scale dependent
            a = pm.Normal("a", mu=0, sigma=3, dims='splines')
            # TEST a = pm.Normal("a", mu=0, sigma=3) # a is scale dependent

            # set priors on break points
            if self.prior['distribution'] == 'normal':
                hs = self.set_normal_prior()
            elif self.prior['distribution'] == 'uniform':
                hs = self.set_uniform_prior()
            else:
                raise NotImplementedError('Prior distribution not implemented')

            # likelihood
            inf = at.constant([np.inf], dtype='float64').reshape((-1, 1))
            hs1 = at.concatenate([hs, inf])
            # could use at.switch instead
            x = at.clip(h - hs, 1e-6, hs1[1:] - hs)
            # TEST b = at.exp(w * at.log(x.T))
            # TEST q = a + at.log(at.sum(b, axis=1))
            b = at.exp(a + w * at.log(x.T))
            q = at.log(at.sum(b, axis=1))

            sigma = pm.HalfCauchy("sigma", beta=0.1)
            obs = pm.Normal('model_q', q, np.sqrt(sigma**2 + q_sigma**2),
                            shape=h.shape, observed=log_q_z)


class BrokenPowerLawRating(PowerLawRating):
    """Experimental segmented power law rating with standard parameterization.

    (See https://en.wikipedia.org/wiki/Power_law#Broken_power_law).
    """
    # Give the model a name
    _model_type = "BrokenPowerLawRating"

    # And a version
    version = "0.1"

    def build_model(self,
                    h: ArrayLike,
                    q: ArrayLike,
                    q_sigma: ArrayLike = None,
                    **kwargs):
        """Creates the PyMC model.

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

            # observations
            h = pm.MutableData("h", self.h_obs)
            log_q_z = pm.MutableData("log_q_z", self.log_q_z)
            q_sigma = pm.MutableData("q_sigma", self.q_sigma)

            # Priors
            # alpha, the power law slopes
            # lower bound of truncated normal forces discharge to increase
            #   with stage
            alpha = pm.Uniform("alpha", lower=0, upper=100,
                               shape=(self.segments, 1), dims="splines")
            # a is the scale parameter
            a = pm.Uniform("a", lower=-100, upper=100)
            sigma = pm.HalfCauchy("sigma", beta=0.1)

            # Set priors on break points
            if self.model_config.get('prior').get('distribution',
                                                  'uniform') == 'normal':
                hs = self.set_normal_prior()
            elif self.model_config.get('prior').get('distribution',
                                                    'uniform') == 'uniform':
                hs = self.set_uniform_prior()
            else:
                raise NotImplementedError('Prior distribution not '
                                          'implemented.')

            # -1 gives alpha_{i-1) - alpha_i rather than
            #   alpha_i - alpha_{i-1} of diff
            alpha_diff = -1 * at.diff(alpha, axis=0)
            sums = at.cumsum(alpha_diff * at.log(hs[1:]), axis=0)
            # Sum for first element is 0, as it does not have a summation
            sums = at.concatenate([pm.math.constant(0, ndim=2), sums])

            # Create ranges for each segment
            segments_range = at.concatenate([pm.math.constant(0, ndim=2),
                                             hs[1:],
                                             pm.math.constant(np.inf, ndim=2)])

            # Tensors are broadcasts for vectorized computation.
            #   Calculates function within range sets value to 0 everywhere
            #   else. Then sum along segment dimension to collapse.
            q = at.switch(((h - hs[0]) > segments_range[:-1]) &
                          ((h - hs[0]) <= segments_range[1:]),
                          a + alpha * at.log(h - hs[0]) + sums, 0)
            q = at.sum(q, axis=0)

            obs = pm.Normal('model_q', q, np.sqrt(sigma**2 + q_sigma**2),
                            shape=h.shape, observed=log_q_z)


class SmoothlyBrokenPowerLawRating(BrokenPowerLawRating):
    """Experimental smooothly broken segmented power law rating

    (See https://en.wikipedia.org/wiki/Power_law#Smoothly_broken_power_law).
    """
    # Give the model a name
    _model_type = "SmoothlyBrokenPowerLawRating"

    # And a version
    version = "0.1"

    def build_model(self,
                    h: ArrayLike,
                    q: ArrayLike,
                    q_sigma: ArrayLike = None,
                    **kwargs):
        """Creates the PyMC model.

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

            # observations
            h = pm.MutableData("h", self.h_obs)
            log_q_z = pm.MutableData("log_q_z", self.log_q_z)
            q_sigma = pm.MutableData("q_sigma", self.q_sigma)

            # Priors
            # alpha, the power law slopes
            # lower bound of truncated normal forces discharge to increase
            #   with stage
            alpha = pm.Uniform("alpha", lower=0, upper=100,
                               shape=(self.segments, 1), dims="splines")
            # a is the scale parameter
            a = pm.Uniform("a", lower=-100, upper=100)
            # delta is the smoothness parameter, limit lower bound (m) to
            #   prevent floating point errors
            delta = pm.Pareto('delta', alpha=0.5, m=0.01)
            sigma = pm.HalfCauchy("sigma", beta=0.1)

            # Set priors on break points
            if self.model_config.get('prior').get('distribution',
                                                  'uniform') == 'normal':
                hs = self.set_normal_prior()
            elif self.model_config.get('prior').get('distribution',
                                                    'uniform') == 'uniform':
                hs = self.set_uniform_prior()
            else:
                raise NotImplementedError('Prior distribution not '
                                          'implemented')

            alpha_diff = at.diff(alpha, axis=0)
            sum_array = (alpha_diff * delta) * \
                at.log(1 + ((h - hs[0])/hs[1:]) ** (1/delta))
            sums = at.sum(sum_array, axis=0)

            obs = pm.Normal("model_q",
                            a + at.log(h - hs[0]) * alpha[0, ...] + sums,
                            np.sqrt(sigma**2 + q_sigma**2), shape=h.shape, observed=log_q_z)
