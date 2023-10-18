"""Modifies PyMC ModelBuilder to work with the Rating model class"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
from pathlib import Path
import json
import math
import warnings

from .transform import LogZTransform
from .model_builder import ModelBuilder

if TYPE_CHECKING:
    from arviz import InferenceData
    from numpy.typing import ArrayLike
    from pymc.util import RandomState
    from typing import Any, Optional


class RatingModelBuilder(ModelBuilder):
    """Parent class for other rating models.

    Sets not implemented PyMC ModelBuilder class functions. Additionally,
    tweaks other ModelBuilder functions for better application in rating curve
    fitting.
    """

    def __init__(self, **kwargs):
        """Update `ModelBuilder.__init__` to only configure the model.

        Configuring the sampler now occurs in the `.fit()` call.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to the model configuration.
        """
        model_config = self.get_default_model_config(**kwargs)
        self.model_config = model_config  # parameters for priors etc.

        self.model = None  # Set by build_model
        # idata is generated during fitting
        self.idata: Optional[az.InferenceData] = None
        self.is_fitted_ = False

    def _data_setter(self,
                     h: ArrayLike,
                     q: ArrayLike = None,
                     q_sigma: ArrayLike = None):
        """Sets new data in the model.

        Parameters
        ----------
        h : array_like
            Input training array of gage height (h) observations.
        q : array_like, optional
            Target discharge (q) values.
        q_sigma : array_like, optional
            Discharge uncertainty in units of discharge.
        """
        with self.model:
            pm.set_data({"h": np.array(h)})
            if q is not None:
                pm.set_data({"log_q_z": self.q_transform.transform(np.array(q))})
            if q_sigma is not None:
                # Approximate sigma as a geometric error
                pm.set_data({"q_sigma":
                             np.log(1 + np.array(q_sigma)/np.array(q))})

            # Set q_sigma = 0 by default
            elif len(self.q_sigma.shape) != 0:
                pm.set_data({"q_sigma": np.zeros(len(h))})

    @property
    def output_var(self) -> str:
        """Name of the output of dependent variable."""
        return "model_q"

    def get_default_sampler_config(self,
                                   n: int = 200_000,
                                   abs_tol: float = 2e-3,
                                   rel_tol: float = 2e-3,
                                   adam_learn_rate: float = 0.001,
                                   draws: int = None,
                                   tune: int = 2_000,
                                   chains: int = 4,
                                   target_accept: float = 0.95,
                                   **kwargs) -> dict:
        """Create sampler configuration dictionary.

        Generates a `sampler_config` dictionary with all the required
        sampler configuration parameters needed to sample/fit the model. It
        will be passed to the class instance when fitting. Any sampler
        parameters not specified to the `.fit()` call, will be set to the
        defaults.

        Parameters
        ----------
        n : int
            The number of iterations. (Only used in ADVI algorithm.)
        abs_tol : float
           Convergence criterion for algorithm. Termination of fitting
           occurs when the absolute tolerance between two consecutive
           iterates is at most `abs_tol`. (Only used in ADVI algorithm.)
        rel_tol : float
           Convergence criterion for algorithm. Termination of fitting
           occurs when the relative tolerance between two consecutive
           iterates is at most `rel_tol`. (Only used in ADVI algorithm.)
        adam_learn_rate : float
            The learning rate for the ADAM Optimizer. (Only used in ADVI
            algorithm.)
        draws : int
            The number of samples to draw. (Used in both algorithms.)
        tune : int
            Number of iterations to tune. Samplers adjust the step sizes,
            scalings or similar during tuning. Tuning samples are discarded.
            (Only used in NUTS algorithm.)
        chains : int
            The number of chains to sample. (Only used in NUTS algorithm.)
        target_accept : float
            The step size is tuned such that we approximate this acceptance
            rate. (Only used in NUTS algorithm.)

        Returns
        -------
        sampler_config : dict
            A dictionary containing all the required sampler configuration
            parameters.
        """
        if self.method == "advi":
            if draws is None:
                draws = 10_000
            sampler_config = {"n": n, 'abs_tol': abs_tol, 'rel_tol': rel_tol,
                              'adam_learn_rate': adam_learn_rate,
                              'draws': draws}
        elif self.method == "nuts":
            if draws is None:
                draws = 1_000
            sampler_config = {"draws": draws, "tune": tune,
                              "chains": chains, "target_accept": target_accept}

        return sampler_config

    def _generate_and_preprocess_model_data(self,
                                            h: ArrayLike,
                                            q: ArrayLike,
                                            q_sigma: ArrayLike):
        """Pre-process input data before fitting the model.

        Pre-processing consists of converting inputs to flattened arrays and
        tranforming data as needed.

        Parameters
        ----------
        h : array_like
            Input training array of gage height (h) observations.
        q : array_like
            Target discharge (q) values.
        q_sigma : array_like
            Discharge uncertainty in units of discharge.
        """
        # Convert inputs to numpy arrays
        self.h_obs = np.array(h).flatten()
        self.q_obs = np.array(q).flatten()

        # Make sure discharge is positive
        if np.any(self.q_obs <= 0):
            raise ValueError('Discharge must be positive. Zero values may be'
                             'allowed in a future release.')

        # We want to fit in log space, so do that pre-processing here.
        # Also, we want to normalize discharge
        self.q_transform = LogZTransform(self.q_obs)
        self.log_q_z = self.q_transform.transform(self.q_obs)

        # transform observational uncertainty to log scale
        if q_sigma is None:
            self.q_sigma = np.array(0)
        else:
            self.q_sigma = np.log(1 + np.array(q_sigma).flatten()/self.q_obs)

        # Set coordinates. Note that self.segments will need to be set in
        # `build_model` before this function is called.
        self.model_coords = {'obs': np.arange(len(self.q_obs)),
                             'splines': np.arange(self.segments)}

    def sample_model(self, **kwargs) -> InferenceData:
        """Update `ModelBuilder.sample_model` with other fitting algorithms.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the PyMC sampler.

        Returns
        -------
        self : InferenceData
            Arviz InferenceData object containing posterior samples of model
            parameters of the fitted model.
        """
        if self.model is None:
            raise RuntimeError(
                "The model hasn't been built yet, call .build_model() first"
                "or call .fit() instead.")

        # Allow for using of the ADVI fitting algorithm
        with self.model:
            if self.method == "advi":
                # Need to fully set up sampler configuration, since non-
                # numerical/string values (i.e., callbacks and optimizer)
                # cannot be serialized and saved to idata
                cb = [pm.callbacks.CheckParametersConvergence(
                          tolerance=self.sampler_config.get('abs_tol'),
                          diff="absolute"),
                      pm.callbacks.CheckParametersConvergence(
                          tolerance=self.sampler_config.get('rel_tol'),
                          diff="relative"),
                      ]

                sampler_config = {'n': self.sampler_config.get('n'),
                                  'obj_optimizer': pm.adam(
                  learning_rate=self.sampler_config.get('adam_learn_rate')),
                                  'callbacks': cb
                                  }
                sampler_args = {**sampler_config, **kwargs}
                # Remove basic string keys from sampler_args dict as we do
                #   not need them now.
                for key in ['abs_tol', 'rel_tol', 'adam_learn_rate', 'draws']:
                    _ = sampler_args.pop(key, None)

                approx = pm.fit(method=pm.ADVI(), **sampler_args)
                idata = approx.sample(draws=self.sampler_config.get('draws'))

            elif self.method == "nuts":
                sampler_args = {**self.sampler_config, **kwargs}
                idata = pm.sample(**sampler_args)

            else:
                raise ValueError(f"Method {self.method} not supported")

            idata.extend(pm.sample_prior_predictive())
            idata.extend(pm.sample_posterior_predictive(idata))

        idata = self.set_idata_attrs(idata)
        return idata

    @classmethod
    def load(cls, fname: str):
        """Update `ModelBuilder.load()` to accept q_sigma.

        ModelBuilder takes two inputs: x and y, so redefine it to accept sigma.
        Creates a ModelBuilder instance from a file and loads inference data
        for the model.

        Parameters
        ----------
        fname : str
            File name with path to saved model.

        Returns
        -------
        An instance of ModelBuilder.
        """
        filepath = Path(str(fname))
        idata = az.from_netcdf(filepath)
        # needs to be converted, because json.loads was changing tuple to list
        model_config = cls._model_config_formatting(json.loads(
                                               idata.attrs["model_config"]))
        model = cls(
            model_config=model_config,
            sampler_config=json.loads(idata.attrs["sampler_config"]),
        )
        model.idata = idata
        dataset = idata.fit_data.to_dataframe()

        # Have loaded fit data include q_sigma and new names
        h = dataset['h']
        q = dataset[model.output_var]
        q_sigma = dataset['q_sigma']
        model.build_model(h, q, q_sigma=q_sigma)

        # All previously used data is in idata.
        if model.id != idata.attrs["id"]:
            raise ValueError(
                f"The file '{fname}' does not contain an inference data of "
                "the same model or configuration as '{cls._model_type}'"
            )

        return model

    def fit(self,
            h: ArrayLike,
            q: ArrayLike,
            q_sigma: ArrayLike = None,
            method: str = 'advi',
            progressbar: bool = True,
            random_seed: RandomState = None,
            **kwargs: Any,
            ) -> InferenceData:
        """Update `ModelBuilder.fit` to accept q_sigma.

        ModelBuilder takes two inputs: x and y, so redefine it to accept sigma.
        Fit a model using the data and algorithm passed as a parameter.
        Sets attrs to inference data of the model.

        Parameters
        ----------
        h : array_like
            Input training array of gage height (h) observations.
        q : array_like
            Target discharge (q) values.
        q_sigma : array_like, optional
            Discharge uncertainty in units of discharge.
        method : str, optional
            The method (algorithm) used to fit the data, options are 'advi'
            or 'nuts'.
        progressbar : bool, optional
            Specifies whether the fit progressbar should be displayed.
        random_seed : RandomState, optional
            Provides sampler with initial random seed for obtaining
            reproducible samples.
        **kwargs : dict
            Algorithm settings can be provided in form of keyword arguments.

        Returns
        -------
        self : InferenceData
            Arviz InferenceData object containing posterior samples of model
            parameters of the fitted model.
        """
        # Save the fitting algorithm method
        self.method = method

        # Define the sampler configuration
        sampler_config = self.get_default_sampler_config(**kwargs)
        self.sampler_config = sampler_config

        # Add progressbar and random_seed kwargs to sampler configuration
        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)

        # Build rating curve models which can include discharge uncertainty
        self.build_model(h, q, q_sigma=q_sigma)

        # Sample (fit) the model
        self.idata = self.sample_model(**sampler_config)

        # Have fit data include uncertainty and have appropriate names
        h_df = pd.DataFrame({'h': h})
        q_df = pd.DataFrame({self.output_var: q})
        if q_sigma is None:
            q_sigma_df = pd.DataFrame({'q_sigma': np.zeros(len(q))})
        else:
            q_sigma_df = pd.DataFrame({'q_sigma': q_sigma})
        combined_data = pd.concat([h_df, q_df, q_sigma_df], axis=1)

        assert all(combined_data.columns), "All columns must have " + \
                                           "non-empty names"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in "
                        "the InferenceData scheme",
            )
            self.idata.add_groups(fit_data=combined_data.to_xarray())

        return self.idata  # type: ignore

    def sample_prior_predictive(self,
                                X_pred,
                                y_pred=None,
                                samples: int = None,
                                extend_idata: bool = False,
                                combined: bool = True,
                                **kwargs,
                                ):
        """Update `ModelBuilder.sample_prior_predicitve` to output untransformed q.
        """
        if y_pred is None:
            y_pred = np.zeros(len(X_pred))
        if samples is None:
            samples = self.sampler_config.get("draws", 500)

        if self.model is None:
            self.build_model(X_pred, y_pred)

        self._data_setter(X_pred, y_pred)
        if self.model is not None:
            with self.model:  # sample with new input data
                prior_pred = pm.sample_prior_predictive(samples, **kwargs)
                self.set_idata_attrs(prior_pred)
                if extend_idata:
                    if self.idata is not None:
                        self.idata.extend(prior_pred)
                    else:
                        self.idata = prior_pred

        prior_predictive_samples = az.extract(prior_pred,
                                              "prior_predictive",
                                              combined=combined)

        return self.q_transform.untransform(prior_predictive_samples)

    def sample_posterior_predictive(self,
                                    X_pred,
                                    extend_idata,
                                    combined,
                                    **kwargs):
        """Update `ModelBuilder.sample_posterior_predicitve` to output untransformed q.
        """
        self._data_setter(X_pred)

        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(self.idata, **kwargs)
            if extend_idata:
                self.idata.extend(post_pred)

        posterior_predictive_samples = az.extract(
            post_pred, "posterior_predictive", combined=combined
        )

        return self.q_transform.untransform(posterior_predictive_samples)

    @property
    def _serializable_model_config(self) -> dict:
        """Return dictionary of model parameters to save.

        `_serializable_model_config` is a property that returns a dictionary
        with all the model parameters that we want to save. As some of the
        data structures are not json serializable, we need to convert them to
        json serializable objects. Some models will need them, others can just
        define them to return the `model_config`.
        """
        return self.model_config

    def predict_posterior(self,
                          X_pred: ArrayLike,
                          extend_idata: bool = True,
                          combined: bool = True,
                          **kwargs,
                          ) -> ArrayLike:
        """Update `ModelBuilder.predict_posterior` to remove data validation.

        Generate posterior predictive samples on unseen data. Exclude any
        data validation as it requires X_pred to be a 2D array-like object.

        Parameters
        ---------
        X_pred : array_like
            The input data used for prediction.
        extend_idata : bool, optional
            Determines whether the predictions should be added to inference
            data object. Defaults to True.
        combined: bool, optional
            Combine chain and draw dims into sample. Won't work if a dim
            named sample already exists. Defaults to True.
        **kwargs: dict
            Additional arguments to pass to `pymc.sample_posterior_predictive`.

        Returns
        -------
        y_pred : ndarray
            Posterior predictive samples for each input X_pred. Shape of array
            is (n_pred, chains * draws) if combined is True, otherwise
            (chains, draws, n_pred).
        """
        posterior_predictive_samples = self.sample_posterior_predictive(
            X_pred, extend_idata, combined, **kwargs
        )

        if self.output_var not in posterior_predictive_samples:
            raise KeyError(
                f"Output variable {self.output_var} not found in posterior "
                "predictive samples."
            )

        # Have this return the data only vs idata DataSet
        return posterior_predictive_samples[self.output_var].data

    def table(self,
              h: ArrayLike = None,
              step: float = 0.01,
              extend: float = 1.1) -> pd.DataFrame:
        """Return stage-discharge rating table.

        Parameters
        ----------
        h : array_like, optional
            Stage values to compute rating table. If None, then use the
            range of observations.
        step : float, optional
            Step size for stage values.
        extend : float, optional
            Extend range of discharge values by this factor.

        Returns
        -------
        rating_table : DataFrame
            Rating table with columns `stage`, mean `discharge`, `median`
            discharge, and `gse` (geometric standard error [1]).

        References
        ----------
        .. [1] Kirkwood, T. B., "Geometric means and measures of dispersion",
               Biometrics, vol. 35, pp. 908-909, 1979
        """
        # Generate stage values if none are input
        if h is None:
            start = self.h_obs.min() - math.remainder(self.h_obs.min(), step)
            stop = self.h_obs.max() * extend + (step - math.remainder(
                                            self.h_obs.max() * extend, step))
            h = np.arange(start, stop, step)

        ratingdata = self.predict_posterior(np.array(h), extend_idata=False)

        df = pd.DataFrame({'stage': h,
                           'discharge': np.mean(ratingdata, axis=1),
                           'median': np.median(ratingdata, axis=1),
                           'gse': np.exp(np.std(np.log(ratingdata),
                                                axis=1))})

        discharge_limit = self.q_obs.max() * extend
        return df[df['discharge'] <= discharge_limit]

    def residuals(self) -> ArrayLike:
        """Compute residuals of rating model.

        Returns
        -------
        residuals : array_like
            Log residuals of rating model.
        """
        # model_builder.ModelBuilder.predict() calls RatingModelBuilder.sample_posterior_predictive()
        # which transforms the data back to the original scale.
        q_pred = self.predict(self.h_obs)

        return np.array(np.log(self.q_obs) - np.log(q_pred))
