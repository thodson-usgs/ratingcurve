"""Plotting functions"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import arviz as az

if TYPE_CHECKING:
    from pandas import DataFrame
    from matplotlib.pyplot import Axes


DEFAULT_FIGSIZE = (5, 5)
NARROW_LINE = 1
REGULAR_LINE = NARROW_LINE * 1.5


class RatingMixin:
    """Parent class for other rating-related mixins."""
    def summary(self, var_names: list=None) -> DataFrame:
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
        if self.idata is None:
            raise AttributeError('Summary cannot be retrieved as model has not been fit.')
            
        return az.summary(self.idata, var_names)



class PlotMixin(RatingMixin):
    """Mixin class for plotting rating models."""
    @staticmethod
    def setup_plot(ax: Axes=None):
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

    
    def plot(self, ax: Axes=None) -> None:
        """Plots gagings and fit rating curve.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        """
        if self.idata is None:
            raise AttributeError('Fitted rating curve cannot be plotted as model has not been fit.')

        ax = self.setup_plot(ax=ax)
        self._format_rating_plot(ax)

        rating_table = self.table()
        h = rating_table['stage']
        q = rating_table['discharge']
        sigma = rating_table['gse']
        ax.plot(q, h, color='black', lw=NARROW_LINE)
        q_u = q * (sigma)**1.96  # this should be 2 sigma
        q_l = q / (sigma)**1.96
        ax.fill_betweenx(h, x1=q_u, x2=q_l, color='lightgray')

        self.plot_gagings(ax=ax)

    
    def plot_residuals(self, ax: Axes=None) -> None:
        """Plots residuals between model and data.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        """
        if self.idata is None:
            raise AttributeError('Rating curve residuals cannot be plotted as model has not been fit.')

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

    
    def plot_gagings(self, ax: Axes=None) -> None:
        """Plot observed gagings with uncertainty.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        """
        if self.idata is None:
            raise AttributeError('Observed gagings cannot be plotted as observed data is given during fitting.')

        ax = self.setup_plot(ax=ax)

        sigma = 1.96 * (np.exp(self.q_sigma) - 1)*np.abs(self.q_obs)

        ax.errorbar(y=self.h_obs, 
                    x=self.q_obs, 
                    xerr=sigma, 
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
    """Mixin class for plotting spline rating models."""
    def plot(self, ax: Axes=None) -> None:
        """Plots spline rating curve.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        """
        if self.idata is None:
            raise AttributeError('Fitted rating curve cannot be plotted as model has not been fit.')

        ax = self.setup_plot(ax=ax)
        self._plot_knots(ax=ax)
        super().plot(ax=ax)

    
    def _plot_knots(self, ax: Axes) -> None:
        """Plots spline knots.

        Parameters
        ----------
        ax : Axes
            Pre-defined matplotlib axes.
        """
        [ax.axhline(k, color='lightgray', linestyle='dotted', linewidth=NARROW_LINE) for k in self._dmatrix.knots]


class PowerLawPlotMixin(PlotMixin):
    """Mixin class for plotting power law rating models."""
    def plot(self, ax: Axes=None) -> None:
        """Plots power-law rating curve.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        """
        if self.idata is None:
            raise AttributeError('Fitted rating curve cannot be plotted as model has not been fit.')

        ax = self.setup_plot(ax=ax)
        self._plot_transitions(ax=ax)
        super().plot(ax=ax)

    
    def plot_residuals(self, ax: Axes=None) -> None:
        """Plots power-law residuals.

        Parameters
        ----------
        ax : Axes, optional
            Pre-defined matplotlib axes.
        """
        if self.idata is None:
            raise AttributeError('Fitted rating curve cannot be plotted as model has not been fit.')

        ax = self.setup_plot(ax=ax)
        self._plot_transitions(ax=ax)
        super().plot_residuals(ax=ax)

    
    def _plot_transitions(self, ax: Axes) -> None:
        """Plot power-law transitions (breakpoints).

        Parameters
        ----------
        ax : Axes
            Defined matplotlib axes.
        """
        hs = self.idata.posterior['hs']

        alpha = 0.05
        hs_u = hs.mean(dim=['chain', 'draw']).data
        hs_lower = hs.quantile(alpha/2, dim=['chain', 'draw']).data.flatten()
        hs_upper = hs.quantile(1 - alpha/2, dim=['chain', 'draw']).data.flatten()

        [ax.axhspan(l, u, color='whitesmoke') for u, l in zip(hs_lower, hs_upper)]
        [ax.axhline(u, color='lightgray', linestyle='dotted', linewidth=NARROW_LINE) for u in hs_u]
