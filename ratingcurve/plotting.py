"""Plotting functions"""

import numpy as np
import matplotlib.pyplot as plt


def plot_spline_rating(model, trace, colors=('tab:blue', 'tab:orange'), ax=None):
    """Plots sline power law rating model

    Returns
    -------
    figure, axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 5))

    q_obs = model.q_obs
    h_obs = model.h_obs

    if model.q_sigma is not None:
        q_sigma = model.q_sigma
    else:
        q_sigma = None

    _plot_gagings(h_obs, q_obs, q_sigma, ax=ax)

    rating_table = model.table(trace)
    _plot_rating(rating_table, ax=ax)

    ax.set_ylabel('Stage')
    ax.set_xlabel('Discharge')


def plot_power_law_rating(model, trace, colors=('tab:blue', 'tab:orange'), ax=None):
    """Plots segmented power law rating model

    Parameters
    ----------
    model : pymc model object
    trace : trace returned by model
    h_obs :
    q_obs :
    colors : list with 2 colornames

    Returns
    -------
    figure, axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 5))

    q_obs = model.q_obs
    h_obs = model.h_obs

    if model.q_sigma is not None:
        q_sigma = model.q_sigma
    else:
        q_sigma = None

    _plot_transitions(trace.posterior['hs'], ax=ax)

    _plot_gagings(h_obs, q_obs, q_sigma, ax=ax)

    rating_table = model.table(trace)
    _plot_rating(rating_table, ax=ax)

    # label
    ax.set_ylabel('Stage')
    ax.set_xlabel('Discharge')


def _plot_transitions(hs, ax=None):
    alpha = 0.05
    hs_u = hs.mean(dim=['chain', 'draw']).data
    hs_lower = hs.quantile(alpha/2, dim=['chain', 'draw']).data.flatten()
    hs_upper = hs.quantile(1 - alpha/2, dim=['chain', 'draw']).data.flatten()

    [ax.axhspan(l, u, color='whitesmoke') for u, l in zip(hs_lower, hs_upper)]
    [ax.axhline(u, color='grey', linestyle='dotted') for u in hs_u]


def _plot_gagings(h_obs, q_obs, q_sigma=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 5))

    if q_sigma is not None:
        sigma_2 = 1.96 * (np.exp(q_sigma) - 1)*q_obs

    else:
        sigma_2 = 0
        # q_sigma = q_sigma * 1.96

    ax.errorbar(y=h_obs, x=q_obs, xerr=sigma_2, fmt="o")


def _plot_rating(discharge_table, ax=None):
    '''TODO Revise
    This function is hack. Should be able to generate posterior predictions directly,
    but this version of pymc seems to have bug. Revisit.
    '''

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 5))

    h = discharge_table['stage']
    q = discharge_table['discharge']
    sigma = discharge_table['sigma']
    ax.plot(q, h, color='black')
    q_u = q * (sigma)**1.96  # this should be 2 sigma
    q_l = q / (sigma)**1.96
    ax.fill_betweenx(h, x1=q_u, x2=q_l, color='lightgray')
