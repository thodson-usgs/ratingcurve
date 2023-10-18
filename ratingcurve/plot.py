"""Plotting functions"""
from __future__ import annotations
from typing import TYPE_CHECKING

import functools
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes
    from pandas import DataFrame


DEFAULT_FIGSIZE = (5, 5)
NARROW_LINE = 1
REGULAR_LINE = NARROW_LINE * 1.5

def is_fit(func):
    """Decorator checks whether model has been fit.
    """
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        if self.idata is None:
            raise RuntimeError(
                "The model hasn't been fit yet, call .fit().")
        return func(self, *args, **kwargs)
    return inner

class RatingMixin:
    """Parent class for other rating-related mixins.
    """
    @is_fit
    def summary(self, var_names: list = None) -> DataFrame:
        """Summary of rating model parameters.

        Parameters
        ----------
        var_names : list of str, optional
            List of variables to include in summary. If no names are given,
            then a summary of all variables is returned.

        Returns
        -------
        df : DataFrame
            DataFrame summary of rating model parameters.
        """
        return az.summary(self.idata, var_names)


class PlotMixin(RatingMixin):
    """Mixin class for plotting rating models.
    """
    @staticmethod
    def setup_plot(ax: Axes = None):
        """Sets up figure and axes for rating curve plot.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.

        Returns
        -------
        ax : Axes
            Generated matplotlib axes.
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=DEFAULT_FIGSIZE)

        return ax

    @is_fit
    def plot(self, ax: Axes = None) -> None:
        """Plots gagings and fit rating curve.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        """
        ax = self.setup_plot(ax=ax)
        self._format_rating_plot(ax)

        rating_table = self.table()
        h = rating_table['stage']
        q = rating_table['discharge']
        sigma = rating_table['gse']
        ax.plot(q, h, color='black', lw=NARROW_LINE)

        # 95% confidence interval
        q_u = q * (sigma)**1.96
        q_l = q / (sigma)**1.96
        ax.fill_betweenx(h, x1=q_u, x2=q_l, color='lightgray')

        self.plot_gagings(ax=ax)

    @is_fit
    def plot_residuals(self, ax: Axes = None) -> None:
        """Plots residuals between model and data.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        """
        ax = self.setup_plot(ax=ax)
        self._format_rating_plot(ax)

        # approximate percentage error
        residuals = self.residuals() * 100
        ax.errorbar(y=self.h_obs,
                    x=residuals,
                    xerr=self.q_sigma*2*100,
                    fmt="o",
                    lw=NARROW_LINE,
                    markersize=4,
                    markerfacecolor='none',
                    markeredgecolor='black',
                    ecolor='black')
        self._format_residual_plot(ax)

    @is_fit
    def plot_gagings(self, ax: Axes = None) -> None:
        """Plot gagings with uncertainty.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        """
        ax = self.setup_plot(ax=ax)

        sigma_2 = 1.96 * (np.exp(self.q_sigma) - 1)*np.abs(self.q_obs)

        ax.errorbar(y=self.h_obs,
                    x=self.q_obs,
                    xerr=sigma_2,
                    fmt="o",
                    lw=1,
                    color='black',
                    markersize=4,
                    markerfacecolor='none')

        self._format_rating_plot(ax)

    @staticmethod
    def _format_rating_plot(ax: Axes) -> None:
        """Format rating plot.

        Parameters
        ----------
        ax : Axes
            Defined matplotlib axes.
        """
        ax.set_ylabel('Stage')
        ax.set_xlabel('Discharge')
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    @staticmethod
    def _format_residual_plot(ax: Axes) -> None:
        """Format residual plot.

        Parameters
        ----------
        ax : Axes
            Defined matplotlib axes.
        """
        ax.set_ylabel('Stage')
        ax.set_xlabel('Percentage Error')

        ax.axvline(0, color='gray', linestyle='solid')
        xlim = ax.get_xlim()
        x_max = max(abs(xlim[0]), abs(xlim[1]))
        ax.set_xlim(-x_max, x_max)


class SplinePlotMixin(PlotMixin):
    """Mixin class for plotting spline rating models.
    """
    @is_fit
    def plot(self, ax: Axes = None) -> None:
        """Plots spline rating curve.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        """

        ax = self.setup_plot(ax=ax)
        self._plot_knots(ax=ax)
        super().plot(ax=ax)

    def _plot_knots(self, ax: Axes) -> None:
        """Plot spline knots.

        Parameters
        ----------
        ax : Axes
            Pre-defined matplotlib axes.
        """
        kwargs = {'color' : 'lightgray',
                  'linestyle' : 'dotted',
                  'linewidth' : NARROW_LINE}

        _ = [ax.axhline(k, **kwargs) for k in self._dmatrix.knots]


class PowerLawPlotMixin(PlotMixin):
    """Mixin class for plotting power law rating models.
    """
    @is_fit
    def plot(self, ax: Axes = None) -> None:
        """Plots power-law rating curve.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        """
        ax = self.setup_plot(ax=ax)
        self._plot_transitions(ax=ax)
        super().plot(ax=ax)

    @is_fit
    def plot_residuals(self, ax: Axes = None) -> None:
        """Plots power-law residuals.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        """
        ax = self.setup_plot(ax=ax)
        self._plot_transitions(ax=ax)
        super().plot_residuals(ax=ax)

    @is_fit
    def _plot_transitions(self, ax: Axes) -> None:
        """Plot power-law transitions (breakpoints).

        Parameters
        ----------
        ax : Axes
            Pre-defined matplotlib axes.
        """
        hs = self.idata.posterior['hs']

        alpha = 0.05
        hs_u = hs.mean(dim=['chain', 'draw']).data
        hs_lower = hs.quantile(alpha/2,
                               dim=['chain', 'draw']).data.flatten()
        hs_upper = hs.quantile(1 - alpha/2,
                               dim=['chain', 'draw']).data.flatten()
        # Plot prediction interval
        i_kwargs = {'color' : 'whitesmoke'}
        _ = [ax.axhspan(l, u, **i_kwargs) for u, l in zip(hs_lower, hs_upper)]

        # Plot expectation
        e_kwargs = {'color' : 'lightgray',
                    'linestyle' : 'dotted',
                    'linewidth' : NARROW_LINE}
        _ = [ax.axhline(u, **e_kwargs) for u in hs_u]
