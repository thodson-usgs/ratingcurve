"""Mixins adding sklearn-like functionality.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

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
        return 10_000

    @property
    def __default_nuts_kwargs(self):
        return {'tune': 2000, 'chains':4, 'cores':4, 'target_accept':0.95}

    @property
    def __default_advi_kwargs(self):
        return {
            'n': 200_000,
            'callbacks': [pm.callbacks.CheckParametersConvergence()]
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
 
    def fit(self, method="advi", **kwargs):
        """Fit the model to the data

        Parameters
        ----------
        method : str, optional
            Method to use for inference, by default "advi"

        Raises
        ------
        ValueError
            If method is not supported

        TODO: Add support for model evaluation
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
        inference_kwargs : dict
            Keyword arguments to pass to pm.sample

        """
        nuts_kwargs = self.__default_nuts_kwargs.copy()
        nuts_kwargs.update(kwargs)
        return pm.sample(**nuts_kwargs)

    def save(self, filename: str) -> None:
        """Save model to file
        """
        raise NotImplementedError

    @staticmethod
    def load(filename: str) -> Model:
        """Load a saved model
        """
        raise NotImplementedError
