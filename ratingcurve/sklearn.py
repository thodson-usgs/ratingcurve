"""Mixins adding sklearn-like functionality.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import sys
import warnings

import pymc as pm

from pymc import Model

from .plot import RatingMixin

if TYPE_CHECKING:
    from arviz import InferenceData
    from numpy.typing import ArrayLike


class RegressorMixin(RatingMixin):
    """Mixin class adding sklearn syntax to regression models
    """
    @property
    def __default_advi_draws(self):
        """Default number of draws for ADVI inference"""
        return 10_000

    @property
    def __default_nuts_kwargs(self):
        """Default keyword arguments for NUTS inference"""
        # PyMC has known bug using NUTS with Windows and multiple cores.
        if sys.platform == 'win32':
            cores = 1
        else:
            cores = 4

        return {'tune': 2000, 'chains':4, 'cores':cores, 'target_accept':0.95}

    @property
    def __default_advi_kwargs(self):
        """Default keyword arguments for ADVI inference"""
        cb = [
            pm.callbacks.CheckParametersConvergence(tolerance=2e-3, diff="absolute"),
            pm.callbacks.CheckParametersConvergence(tolerance=2e-3, diff="relative"),
        ]

        return {
            'n': 200_000,
            'obj_optimizer' : pm.adam(learning_rate=.001), # converges faster than adagrad_window
            'callbacks': cb
        }

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict using the model

        Parameters
        ----------
        X : ArrayLike
            Input data

        Returns
        -------
        ArrayLike
            Predicted values
        """
        raise NotImplementedError
 
    def fit(self, method="advi", **kwargs) -> InferenceData:
        """Fit the model to the data

        Parameters
        ----------
        method : str, optional
            Method to use for inference, by default "advi"

        Returns
        -------
        InferenceData
            ArviZ InferenceData object

        Raises
        ------
        ValueError
            If method is not supported
        """
        with self.model:
            if method == "advi":
                trace = self._advi_inference(**kwargs)
            elif method == "nuts":
                trace = self._nuts_inference(**kwargs)
            else:
                raise ValueError(f"Method {method} not supported")
            
        self.trace = trace
        return trace

    def _advi_inference(self, **kwargs):
        """Run ADVI inference

        Pass parameters to ADVI inside the model context.
        
        Parameters
        ----------
        inference_kwargs : dict
            Keyword arguments to pass to pm.fit

        Returns
        -------
        trace : InferenceData
            ArviZ InferenceData object
        kwargs : dict
            Keyword arguments to pass to pm.fit
        """
        advi_kwargs = self.__default_advi_kwargs.copy()
        advi_kwargs.update(kwargs)

        method = pm.ADVI()
        approx = pm.fit(method=method, **advi_kwargs)
        return approx.sample(draws=self.__default_advi_draws)

    def _nuts_inference(self, **kwargs):
        """Run NUTS inference
        
        Pass parameters to NUTS inside the model context.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to pass to pm.sample
        """
        nuts_kwargs = self.__default_nuts_kwargs.copy()
        nuts_kwargs.update(kwargs)

        # Check to see if OS is Windows. If so, set cores=1 and print warning.
        if sys.platform == 'win32':
            warnings.warn("PyMC NUTS sampler has known bug when fitting with "
                          "more than one core on Windows. Fitting with more "
                          "than one core will cause an error.", RuntimeWarning)

        return pm.sample(**nuts_kwargs)

    def save(self, filename: str) -> None:
        """Save model to file
        """
        raise NotImplementedError

    @staticmethod
    def load(filename: str) -> Model:
        """Load a saved model

        Parameters
        ----------
        filename : str
            Path to saved model

        Returns
        -------
        Model
            PyMC3 model
        """
        raise NotImplementedError
