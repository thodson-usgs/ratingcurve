"""Plotting functions"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from .ratingmodel import Rating, PowerLawRating, SplineRating
    from arviz import InferenceData


DEFAULT_FIGSIZE = (5,5)


def _plot_residuals(rating: Rating, trace: InferenceData, ax=None):
    """Plots residuals

    Parameters
    ----------
    rating : Rating
        Rating model
    trace : ArviZ InferenceData
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=DEFAULT_FIGSIZE)

    if rating.q_sigma is not None:
        q_sigma = rating.q_sigma
    else:
        q_sigma = None

    residuals = rating.residuals(trace)

    hs = trace.posterior.get('hs', None)

    if hs is not None:
        _plot_transitions(trace.posterior['hs'], ax=ax)

    _plot_gagings(rating.h_obs, residuals, q_sigma=q_sigma, ax=ax)

    _format_residual_plot(ax)


def plot_spline_rating(rating: SplineRating, trace: InferenceData, ax=None):
    """Plots sline power law rating model

    Parameters
    ----------
    rating : SplineRating
        Spline rating model
    trace : ArviZ InferenceData
    ax : matplotlib axes

    Returns
    -------
    figure, axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=DEFAULT_FIGSIZE)

    q_obs = rating.q_obs
    h_obs = rating.h_obs

    if rating.q_sigma is not None:
        q_sigma = rating.q_sigma
    else:
        q_sigma = None

    _plot_gagings(h_obs, q_obs, q_sigma, ax=ax)

    _plot_rating(rating.table(trace), ax=ax)

    _format_rating_plot(ax)


def plot_power_law_rating(rating: PowerLawRating, trace: InferenceData, ax=None):
    """Plots segmented power law rating model

    Parameters
    ----------
    rating : PowerLawRating
    trace : ArviZ InferenceData
    ax : matplotlib axes

    Returns
    -------
    figure, axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=DEFAULT_FIGSIZE)

    q_obs = rating.q_obs
    h_obs = rating.h_obs

    if rating.q_sigma is not None:
        q_sigma = rating.q_sigma
    else:
        q_sigma = None

    _plot_transitions(trace.posterior['hs'], ax=ax)

    _plot_gagings(h_obs, q_obs, q_sigma, ax=ax)

    _plot_rating(rating.table(trace), ax=ax)

    _format_rating_plot(ax)


def _plot_transitions(hs, ax):
    """Plot transitions (breakpoints)

    Parameters
    ----------
    hs : xarray DataArray
    ax : matplotlib axes
    """
    alpha = 0.05
    hs_u = hs.mean(dim=['chain', 'draw']).data
    hs_lower = hs.quantile(alpha/2, dim=['chain', 'draw']).data.flatten()
    hs_upper = hs.quantile(1 - alpha/2, dim=['chain', 'draw']).data.flatten()

    [ax.axhspan(l, u, color='whitesmoke') for u, l in zip(hs_lower, hs_upper)]
    [ax.axhline(u, color='grey', linestyle='dotted') for u in hs_u]


def _plot_gagings(h_obs, q_obs, q_sigma=None, ax=None):
    """Plot gagings with uncertainty

    Parameters
    ----------
    h_obs : array-like
        Stage observations.
    q_obs : array-like
        Discharge observations.
    q_sigma : array-like, optional
        Discharge uncertainty (1 sigma)
    ax : matplotlib axes, optional
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=DEFAULT_FIGSIZE)

    if q_sigma is not None:
        sigma_2 = 1.96 * (np.exp(q_sigma) - 1)*np.abs(q_obs)

    else:
        sigma_2 = 0

    ax.errorbar(y=h_obs, x=q_obs, xerr=sigma_2, fmt="o")


def _plot_rating(rating_table, ax=None):
    """"Plot rating table with uncertainty

    TODO This function is hack. Should be able to generate posterior predictions directly,
    but this version of pymc seems to have bug.

    Parameters
    ----------
    rating_table : pandas DataFrame
    ax : matplotlib axes, optional
    """

    if ax is None:
        fig, ax = plt.subplots(1, figsize=DEFAULT_FIGSIZE)

    h = rating_table['stage']
    q = rating_table['discharge']
    sigma = rating_table['sigma']
    ax.plot(q, h, color='black')
    q_u = q * (sigma)**1.96  # this should be 2 sigma
    q_l = q / (sigma)**1.96
    ax.fill_betweenx(h, x1=q_u, x2=q_l, color='lightgray')


def _format_rating_plot(ax):
    """Format rating plot

    Parameters
    ----------
    ax : matplotlib axes
    """
    ax.set_ylabel('Stage')
    ax.set_xlabel('Discharge')

    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))


def _format_residual_plot(ax):
    """Format residual plot

    Parameters
    ----------
    ax : matplotlib axes
    """
    ax.set_ylabel('Stage')
    ax.set_xlabel('Percentage Error')

    ax.axvline(0, color='grey', linestyle='solid')
    xlim = ax.get_xlim()
    x_max = max(abs(xlim[0]), abs(xlim[1]))
    ax.set_xlim(-x_max, x_max)