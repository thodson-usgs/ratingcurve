"""Plotting functions"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from arviz import InferenceData
    from pandas import DataFrame


DEFAULT_FIGSIZE = (5, 5)
NARROW_LINE = 1
REGULAR_LINE = NARROW_LINE * 1.5

class RatingMixin:
    """Parent class for other rating-related mixins
    """
    @property
    def model(self):
        raise NotImplementedError

    @property
    def trace(self) -> InferenceData:
        """ArviZ InferenceData object

        Returns
        -------
        ArviZ InferenceData object
        """
        return self.__last_trace
    
    @trace.setter
    def trace(self, value):
        self.__last_trace = value

    def summary(self, trace: InferenceData=None, var_names: list=None) -> DataFrame:
        """Summary of rating model parameters
        
        Parameters
        ----------
        trace : ArviZ InferenceData
        var_names : list of str
            List of variables to include in summary
            
        Returns
        -------
        DataFrame summary of rating model parameters
        """
        if trace is None:
            trace = self.trace
        return az.summary(trace, var_names)


class PlotMixin(RatingMixin):
    """Mixin class for plotting rating models
    """
    @staticmethod
    def setup_plot(ax=None):
        """Plots rating curve

        Parameters
        ----------
        ax : matplotlib axes

        Returns
        -------
        ax : matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=DEFAULT_FIGSIZE)

        return ax
 
    def plot(self, trace: InferenceData=None, ax=None) -> None:
        """Plots rating curve

        Parameters
        ----------
        trace : ArviZ InferenceData
        ax : matplotlib axes
        """
        if trace is None:
            trace = self.trace

        ax = self.setup_plot(ax=ax)
        self._format_rating_plot(ax)

        rating_table = self.table(trace)
        h = rating_table['stage']
        q = rating_table['discharge']
        sigma = rating_table['gse']
        ax.plot(q, h, color='black', lw=NARROW_LINE)
        q_u = q * (sigma)**1.96  # this should be 2 sigma
        q_l = q / (sigma)**1.96
        ax.fill_betweenx(h, x1=q_u, x2=q_l, color='lightgray')

        self.plot_gagings(ax=ax)

    def plot_residuals(self, trace: InferenceData=None, ax=None) -> None:
        """Plots residuals

        Parameters
        ----------
        trace : ArviZ InferenceData
        ax : matplotlib axes
        """
        if trace is None:
            trace = self.trace

        ax = self.setup_plot(ax=ax)
        self._format_rating_plot(ax)

        # TODO: this could be a function
        if self.q_sigma is not None:
            q_sigma = self.q_sigma
        else:
            q_sigma = None

        # approximate percentage error
        residuals = self.residuals(trace) * 100
        residuals = residuals.mean(axis=1)
        ax.errorbar(y=self.h_obs, x=residuals, xerr=q_sigma*2*100, fmt="o",
                    lw=NARROW_LINE,
                    markersize=4,
                    markerfacecolor='none',
                    markeredgecolor='black',
                    ecolor='black')
        self._format_residual_plot(ax)

    def plot_gagings(self, ax=None) -> None:
        """Plot gagings with uncertainty

        Parameters
        ----------
        ax : matplotlib axes, optional
        """
        ax = self.setup_plot(ax=ax)

        # TODO: safely get these
        q_sigma = self.q_sigma
        h_obs = self.h_obs
        q_obs = self.q_obs

        if q_sigma is not None:
            sigma_2 = 1.96 * (np.exp(q_sigma) - 1)*np.abs(q_obs)

        else:
            sigma_2 = 0

        ax.errorbar(y=h_obs, x=q_obs, xerr=sigma_2, fmt="o", lw=1,
                    color='black',
                    markersize=4,
                    markerfacecolor='none')
        self._format_rating_plot(ax)

    @staticmethod
    def _format_rating_plot(ax) -> None:
        """Format rating plot

        Parameters
        ----------
        ax : matplotlib axes
        """
        ax.set_ylabel('Stage')
        ax.set_xlabel('Discharge')
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    @staticmethod
    def _format_residual_plot(ax) -> None:
        """Format residual plot

        Parameters
        ----------
        ax : matplotlib axes
        """
        ax.set_ylabel('Stage')
        ax.set_xlabel('Percentage Error')

        ax.axvline(0, color='gray', linestyle='solid')
        xlim = ax.get_xlim()
        x_max = max(abs(xlim[0]), abs(xlim[1]))
        ax.set_xlim(-x_max, x_max)


class SplinePlotMixin(PlotMixin):
    """Mixin class for plotting spline rating models
    """
    def plot(self, trace: InferenceData=None, ax=None):
        """Plots rating curve

        Parameters
        ----------
        trace : ArviZ InferenceData
        ax : matplotlib axes
        """
        ax = self.setup_plot(ax=ax)
        self._plot_knots(ax=ax)
        super().plot(trace, ax=ax)

    def _plot_knots(self, ax):
        """Plot spline knots

        Parameters
        ----------
        ax : matplotlib axes
        """
        [ax.axhline(k, color='lightgray', linestyle='dotted', linewidth=NARROW_LINE) for k in self._dmatrix.knots]


class PowerLawPlotMixin(PlotMixin):
    """Mixin class for plotting power law rating models
    """
    def plot(self, trace: InferenceData=None, ax=None):
        """Plots rating curve

        Parameters
        ----------
        trace : ArviZ InferenceData
        ax : matplotlib axes

        TODO: this function might be cleaner as a wrapper around PlotMixin.plot
        """
        if trace is None:
            trace = self.trace

        ax = self.setup_plot(ax=ax)
        self._plot_transitions(trace, ax=ax)
        super().plot(trace, ax=ax)

    def plot_residuals(self, trace: InferenceData=None, ax=None):
        """Plots residuals

        Parameters
        ----------
        trace : ArviZ InferenceData
        ax : matplotlib axes
        """
        if trace is None:
            trace = self.trace

        ax = self.setup_plot(ax=ax)
        self._plot_transitions(trace, ax=ax)
        super().plot_residuals(trace, ax=ax)

    def _plot_transitions(self, trace: InferenceData, ax):
        """Plot transitions (breakpoints)

        Parameters
        ----------
        trace : ArviZ InferenceData
            Inference data containing transition points (hs)
        ax : matplotlib axes
        """
        hs = trace.posterior['hs']

        alpha = 0.05
        hs_u = hs.mean(dim=['chain', 'draw']).data
        hs_lower = hs.quantile(alpha/2, dim=['chain', 'draw']).data.flatten()
        hs_upper = hs.quantile(1 - alpha/2, dim=['chain', 'draw']).data.flatten()

        [ax.axhspan(l, u, color='whitesmoke') for u, l in zip(hs_lower, hs_upper)]
        [ax.axhline(u, color='lightgray', linestyle='dotted', linewidth=NARROW_LINE) for u in hs_u]
