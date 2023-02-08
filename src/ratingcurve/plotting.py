import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import aesara.tensor as at


def plot_spline_rating(model, trace, colors=('tab:blue', 'tab:orange'), ax=None):
    if ax is None:
        fig, axes = plt.subplots(1, figsize=(5, 5))

    q_obs = model.q_obs
    h_obs = model.h_obs
    
    if model.q_sigma is not None:
        q_sigma = model.q_sigma
    else:
        q_sigma = None


    _plot_gagings(h_obs, q_obs, q_sigma, ax=axes)

    rating_table = model.table(trace)
    _plot_rating(rating_table, ax=axes)
    #_plot_spline_rating(trace=trace,
    #                    model=model,
    #                    h_min=h_obs.min(),
    #                    h_max=h_obs.max(),
    #                    ax=axes
    #                   )

    axes.set_ylabel('Stage')
    axes.set_xlabel('Discharge')


def plot_power_law_rating(model, trace, colors=('tab:blue', 'tab:orange'), ax=None):
    """
    Plots segmented power law rating model
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
        fig, axes = plt.subplots(1, figsize=(5,5))

    q_obs = model.q_obs#.flatten()
    h_obs = model.h_obs#.flatten()
    if model.q_sigma is not None:
        q_sigma = model.q_sigma#.flatten()
    else:
        q_sigma = None

    # q_obs = model.q_transform.transform(q_obs)

    _plot_transitions(trace.posterior['hs'],
                      0, q_obs.max(), ax=axes)

    _plot_gagings(h_obs, q_obs, q_sigma, ax=axes)
    
    rating_table = model.table(trace)
    _plot_rating(rating_table, ax=axes)
    #_plot_powerlaw_rating(trace,
    #                      h_min=h_obs.min(),
    #                      h_max=h_obs.max(),
    #                      transform=model.q_transform,
    #                      ax=axes)

    # label
    axes.set_ylabel('Stage')
    axes.set_xlabel('Discharge')


def _plot_transitions(hs, q_min, q_max, ax=None):
    alpha = 0.05
    hs_u = hs.mean(dim=['chain', 'draw']).data.squeeze()
    hs_lower = hs.quantile(alpha/2, dim=['chain', 'draw']).data.squeeze()
    hs_upper = hs.quantile(1 - alpha/2, dim=['chain', 'draw']).data.squeeze()

    [ax.axhspan(l, u, color='whitesmoke') for l, u in zip(hs_lower, hs_upper)]
    [ax.axhline(u, color='grey', linestyle='dotted') for u in hs_u]


def _plot_gagings(h_obs, q_obs, q_sigma=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 5))

    if q_sigma is not None:
        sigma_2 = 1.96 * (np.exp(q_sigma) - 1)*q_obs
    
    else:
        sigma_2 = 0
        #q_sigma = q_sigma * 1.96

    ax.errorbar(y=h_obs, x=q_obs, xerr=sigma_2, fmt="o")


def _plot_rating(discharge_table, ax=None):
    ''' TODO Revise
    This function is hack. Should be able to generate posterior predictions directly,
    but this version of pymc seems to have bug. Revisit.
    '''

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5,5))

    h = discharge_table['stage']
    q = discharge_table['discharge']
    sigma = discharge_table['sigma']
    ax.plot(q, h, color='black')
    q_u = q * (sigma)**1.96 # this should be  2 sigma
    q_l = q / (sigma)**1.96
    ax.fill_betweenx(h, x1=q_u, x2=q_l, color='lightgray')

