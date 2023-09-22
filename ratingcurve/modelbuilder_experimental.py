"""Experimental ModelBuilder Setup"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
from pymc_experimental.model_builder import ModelBuilder
import arviz as az
import pandas as pd
from pathlib import Path
import json
import math
import warnings

if TYPE_CHECKING:
    from arviz import InferenceData
    from numpy.typing import ArrayLike
    from pymc.util import RandomState
    from typing import Any


class RatingModelBuilder(ModelBuilder):
    """
    Parent class for other rating models that sets not implemented 
    PyMC ModelBuilder class functions. Additionally, tweaks fitting and loading
    functions for better application in rating curve fitting.
    """

    def __init__(self,
                 method: str='advi',
                 model_config: dict = None,
                 sampler_config: dict = None,
                 ):
        """
        Updates the ModelBuilder initialization to save the input method.

        Parameters
        ----------
        method : str, optional
            The method (algorithm) used to fit the data, options are 'advi' or 'nuts'.
        model_config : Dictionary, optional
            dictionary of parameters that initialise model configuration. Class-default defined by the user default_model_config method.
        sampler_config : Dictionary, optional
            dictionary of parameters that initialise sampler configuration. Class-default defined by the user default_sampler_config method.
        """
        # Save the fitting algorithm method
        self.method = method
        super().__init__(model_config=model_config, sampler_config=sampler_config)


    def _data_setter(self, h: ArrayLike, q: ArrayLike=None, q_sigma: ArrayLike=None):
        """
        Sets new data in the model.

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
                pm.set_data({"logq": np.log(np.array(q))})
            if q_sigma is not None:
                pm.set_data({"q_sigma": np.log(1 + np.array(q_sigma)/np.array(q))})

            # Need to update q_sigma if it was input. (If not input originally, it would
            # be set to a numpy scalar array = 0. Therefore, it would not need to be updated.)
            # We change it to be a zero array of length len(h) to match the h data. We use zeros
            # as the model has already been fit and parameters estimated using the input q_sigma data.
            # Therefore, predictions (which `_data_setter` is used for) would not need any
            # observational uncertainty as it would already be included in the parameter estimation.
            elif len(self.q_sigma.shape) != 0:
                pm.set_data({"q_sigma": np.zeros(len(h))})

    
    @property
    def output_var(self) -> str:
        """
        Name of the output of dependent variable.
        """
        return "model_q"

    
    def get_default_sampler_config(self) -> dict:
        """
        Returns a `sampler_config` dictionary with all the required sampler configuration parameters
        needed to sample/fit the model. It will be passed to the class instance on
        initialization, in case the user doesn't provide any sampler_config of their own.

        When specified by a user, `model_config` must be a dictionary and contain the certain keys
        depending on the fitting algorithm `method`. For ADVI algorithm (i.e., `method='advi'`), these 
        keys must include and are formatted as follows:

        n : int
            The number of iterations.
        abs_tol : dict
           Convergence criterion for algorithm. Termination of fitting occurs when the
           absolute tolerance between two consecutive iterates is at most `abs_tol`.
        rel_tol : dict
           Convergence criterion for algorithm. Termination of fitting occurs when the
           relative tolerance between two consecutive iterates is at most `rel_tol`.
        adam_learn_rate : dict
            The learning rate for the ADAM Optimizer.
        draws : dict
            The number of samples to draw after fitting.

        For the NUTS algorithm (i.e., `method='nuts'`), these keys must include and are
        formatted as follows:
        
        tune : int
            Number of iterations to tune. Samplers adjust the step sizes, scalings or similar during tuning.
        chains : dict
            The number of chains to sample.
        target_accept : dict
            The step size is tuned such that we approximate this acceptance rate.
        draws : dict
           The number of samples to draw. The number of `tune` samples are discarded. 
        """
        if self.method == "advi":
            sampler_config = {"n": 200_000, 'abs_tol': 2e-3, 'rel_tol': 2e-3, 
                              'adam_learn_rate': 0.001, 'draws': 10_000}
        elif self.method == "nuts":
            sampler_config = {"draws": 1_000, "tune": 2_000, "chains": 4, "target_accept": 0.95}
         
        return sampler_config

    
    def _generate_and_preprocess_model_data(self, h: ArrayLike, q: ArrayLike, q_sigma: ArrayLike):
        """
        Pre-processed the input data before fitting the model. Defines all
        required preprocessing and conditional assignments.
        
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
        
        # We want to fit in log space, so do that pre-processing here.
        self.log_q = np.log(self.q_obs)

        # transform observational uncertainty to log scale
        if q_sigma is None:
            self.q_sigma = np.array(0)
        else:
            self.q_sigma = np.log(1 + np.array(q_sigma).flatten()/self.q_obs)
        
        # Set coordinates. Note that self.segments will need to be set in `build_model` before
        # this function is called.
        self.model_coords = {'obs': np.arange(len(self.q_obs)), 'splines': np.arange(self.segments)}



    def sample_model(self, **kwargs):
        """
        Redefinition of ModelBuilder sample_model function to include other fitting algorithms.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the PyMC sampler.

        Returns
        -------
        self : InferenceData
            Arviz InferenceData object containing posterior samples of model parameters
            of the fitted model.
        """
        if self.model is None:
            raise RuntimeError(
                "The model hasn't been built yet, call .build_model() first or call .fit() instead."
            )

        # Allow for using of the ADVI fitting algorithm
        with self.model:
            if self.method == "advi":
                # Need to fully set up sampler configuration, since non numerical/string values
                # (i.e., callbacks and optimizer) cannot be serialized and saved to idata
                cb = [pm.callbacks.CheckParametersConvergence(tolerance=self.sampler_config.get('abs_tol'), 
                                                              diff="absolute"),
                      pm.callbacks.CheckParametersConvergence(tolerance=self.sampler_config.get('rel_tol'), 
                                                              diff="relative"),
                      ]
    
                sampler_config = {'n': self.sampler_config.get('n'),
                                  'obj_optimizer' : pm.adam(learning_rate=self.sampler_config.get('adam_learn_rate')),
                                  'callbacks': cb
                                  }
                sampler_args = {**sampler_config, **kwargs}
                # Remove basic string keys from sampler_args dict as we do not need them now.
                for key in ['abs_tol', 'rel_tol', 'adam_learn_rate', 'draws']:
                    sampler_args.pop(key, None)
                    
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
        """
        Redefinition of ModelBuilder load function as it needed tweaking from PyMC version.
        Creates a ModelBuilder instance from a file,
        Loads inference data for the model.

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
        model_config = cls._model_config_formatting(json.loads(idata.attrs["model_config"]))
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
                f"The file '{fname}' does not contain an inference data of the same model or configuration as '{cls._model_type}'"
            )

        return model

    
    def fit(self,
            h: ArrayLike,
            q: ArrayLike,
            q_sigma: ArrayLike=None,
            progressbar: bool=True,
            random_seed: RandomState = None,
            **kwargs: Any,
            ) -> az.InferenceData:
        """
        Redefinition of ModelBuilder fit function as it needed tweaking from PyMC version.
        Fit a model using the data passed as a parameter.
        Sets attrs to inference data of the model.

        Parameters
        ----------
        h : array_like
            Input training array of gage height (h) observations.
        q : array_like, optional
            Target discharge (q) values.
        q_sigma : array_like, optional
            Discharge uncertainty in units of discharge.
        progressbar : bool
            Specifies whether the fit progressbar should be displayed.
        random_seed : RandomState
            Provides sampler with initial random seed for obtaining reproducible samples.
        **kwargs : Any
            Custom sampler settings can be provided in form of keyword arguments.

        Returns
        -------
        self : InferenceData
            Arviz InferenceData object containing posterior samples of model parameters
            of the fitted model.
        """
        # Build rating curve models which can include discharge uncertainty
        self.build_model(h, q, q_sigma=q_sigma)

        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)
        self.idata = self.sample_model(**sampler_config)

        # Have fit data include uncertainty and have appropriate names
        h_df= pd.DataFrame({'h': h})
        q_df = pd.DataFrame({self.output_var: q})
        if q_sigma is None:
            q_sigma_df = pd.DataFrame({'q_sigma': np.zeros(len(q))})
        else:
            q_sigma_df = pd.DataFrame({'q_sigma': q_sigma})
        combined_data = pd.concat([h_df, q_df, q_sigma_df], axis=1)

        assert all(combined_data.columns), "All columns must have non-empty names"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )
            self.idata.add_groups(fit_data=combined_data.to_xarray())  # type: ignore

        return self.idata  # type: ignore


    def sample_prior_predictive(self,
                                X_pred,
                                y_pred=None,
                                samples: int=None,
                                extend_idata: bool=False,
                                combined: bool=True,
                                **kwargs,
                                ):
        """
        Update of ModelBuilder `sample_prior_predicitve` function to
        output unlogged discharge (q).
        """
        return np.exp(super().sample_prior_predictive(X_pred, y_pred, samples, extend_idata, combined, **kwargs))
        
        
    def sample_posterior_predictive(self, X_pred, extend_idata, combined, **kwargs):
        """
        Update of ModelBuilder `sample_posterior_predicitve` function to
        output unlogged discharge (q).
        """
        return np.exp(super().sample_posterior_predictive(X_pred, extend_idata, combined, **kwargs))


    @property
    def _serializable_model_config(self) -> dict:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config


    def predict_posterior(self,
                          X_pred: ArrayLike,
                          extend_idata: bool=True,
                          combined: bool=True,
                          **kwargs,
                          ):
        """
        Update of ModelBuilder predict_posterior function to remove data validation check.
        Generate posterior predictive samples on unseen data.

        Parameters
        ---------
        X_pred : array-like if sklearn is available, otherwise array, shape (n_pred, n_features)
            The input data used for prediction.
        extend_idata : Boolean determining whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.
        **kwargs: Additional arguments to pass to pymc.sample_posterior_predictive

        Returns
        -------
        y_pred : ndarray, shape (n_pred, chains * draws) if combined is True, otherwise (chains, draws, n_pred)
            Posterior predictive samples for each input X_pred
        """
        posterior_predictive_samples = self.sample_posterior_predictive(
            X_pred, extend_idata, combined, **kwargs
        )

        if self.output_var not in posterior_predictive_samples:
            raise KeyError(
                f"Output variable {self.output_var} not found in posterior predictive samples."
            )

        # Have this return the data only vs idata DataSet
        return posterior_predictive_samples[self.output_var].data


    def table(self, h: ArrayLike=None, step: float=0.01, extend: float=1.1) -> pd.DataFrame:
        """
        Return stage-discharge rating table.

        Parameters
        ----------
        h : array_like
            Stage values to compute rating table. If None, then use the range of observations.
        step : float
            Step size for stage values.
        extend : float
            Extend range of discharge values by this factor.

        Returns
        -------
        rating_table : DataFrame
            Rating table with columns `stage`, mean `discharge`, `median` discharge, and 
            `gse` (geometric standard error).

        References
        ----------
        .. [1] Kirkwood, T. B., "Geometric means and measures of dispersion",
               Biometrics, vol. 35, pp. 908-909, 1979
        """
        # Generate stage values if none are input
        if h is None:
            start = self.h_obs.min() - math.remainder(self.h_obs.min(), step)
            stop = self.h_obs.max() * extend + (step - math.remainder(self.h_obs.max() * extend, step))
            h = np.arange(start, stop, step)
    
        ratingdata = self.predict_posterior(np.array(h), extend_idata=False)
        rating_table = pd.DataFrame({'stage': h,
                                     'discharge': np.mean(ratingdata, axis=1),
                                     'median': np.median(ratingdata, axis=1),
                                     'gse': np.exp(np.std(np.log(ratingdata), axis=1))})
        # Limit discharge range
        rating_table = rating_table[rating_table['discharge'] <= self.q_obs.max() * extend]

        return rating_table


    def residuals(self) -> ArrayLike:
        """
        Compute residuals of rating model.

        Returns
        -------
        residuals : array_like
            Log residuals of rating model.
        """
        q_pred = self.predict(self.h_obs)

        return np.array(self.log_q - np.log(q_pred))
