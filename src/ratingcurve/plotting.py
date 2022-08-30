import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import aesara.tensor as at


def plot_spline_rating(model, trace, colors=('tab:blue', 'tab:orange'), ax=None):
    if ax is None:
        fig, axes = plt.subplots(1, figsize=(5, 5))

    q_obs = model.q_obs
    h_obs = model.h_obs
    q_sigma = None

    _plot_gagings(h_obs, q_obs, q_sigma, ax=axes)

    _plot_spline_rating(trace=trace,
                        model=model,
                        h_min=h_obs.min(),
                        h_max=h_obs.max(),
                        ax=axes
                       )

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

    q_obs = model.q_obs.flatten()
    h_obs = model.h_obs.flatten()
    if model.q_sigma is not None:
        q_sigma = model.q_sigma.flatten()
    else:
        q_sigma = None

    # q_obs = model.q_transform.transform(q_obs)

    _plot_transitions(trace.posterior['hs'],
                      0, q_obs.max(), ax=axes)

    _plot_gagings(h_obs, q_obs, q_sigma, ax=axes)

    _plot_powerlaw_rating(trace,
                          h_min=h_obs.min(),
                          h_max=h_obs.max(),
                          transform=model.q_transform,
                          ax=axes)

    # label
    axes.set_ylabel('Stage')
    axes.set_xlabel('Discharge')


def _plot_transitions(hs, q_min, q_max, ax=None):
    alpha = 0.05
    hs_u = hs.mean(dim=['chain', 'draw']).data
    hs_lower = hs.quantile(alpha/2, dim=['chain', 'draw']).data
    hs_upper = hs.quantile(1 - alpha/2, dim=['chain', 'draw']).data

    [ax.axhspan(l, u, color='whitesmoke') for u, l in zip(hs_lower, hs_upper)]
    [ax.axhline(u, color='grey', linestyle='dotted') for u in hs_u]


def _plot_gagings(h_obs, q_obs, q_sigma=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 5))

    if q_sigma is not None:
        q_sigma = q_sigma * 1.96

    ax.errorbar(y=h_obs, x=q_obs, xerr=q_sigma, fmt="o")


def _plot_powerlaw_rating(trace, h_min, h_max, transform=None, ax=None):
    ''' TODO Revise
    This function is hack. Should be able to generate posterior predictions directly,
    but this version of pymc seems to have bug. Revisit.
    '''

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5,5))

    a = trace.posterior['a'].values
    w = trace.posterior['w'].values
    hs = trace.posterior['hs'].values

    chain = trace.posterior['chain'].shape[0]
    draw = trace.posterior['draw'].shape[0]

    inf = np.ones((chain, draw, 1)) + np.inf

    clips_array = np.zeros(hs.shape[2])
    clips_array[0] = -np.inf
    clips = clips_array
    # TODO log distribute
    h = np.linspace(h_min, h_max, 100).reshape(-1, 1) #TODO control via parameter
    h_tile = np.tile(h, draw).reshape(-1, draw, chain)

    h0_offset = np.ones(hs.shape[2])
    h0_offset[0] = 0
    h0 = hs - h0_offset

    #b1 = at.switch( at.le(h, hs), clips , at.log(h-h0))
    b1 = np.where(h_tile<=hs, clips, np.log(h_tile-h0))

    mu = a + (b1*w).sum(axis=2)

    if transform:
        mu = transform.untransform(mu)

    alpha = 0.05
    q_mean = mu.mean(axis=1)
    q_u = np.quantile(mu, 1-alpha/2, axis=1) 
    q_l = np.quantile(mu, alpha/2, axis=1)
    ax.plot(q_mean, h, color='black')
    ax.fill_betweenx(h.flatten(), x1=q_u, x2=q_l, color='lightgray')


def _plot_spline_rating(trace, model, h_min, h_max, ax=None):
    ''' TODO
    '''

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5,5))

    w = trace.posterior['w'].values.squeeze()
    chain = trace.posterior['chain'].shape[0]
    draw = trace.posterior['draw'].shape[0]

    h = np.linspace(h_min, h_max, 100) #XXX something wrong here; can't increase h_max
    B = model.d_transform(h)
    mu = np.dot(B, w.T)

    mu = model.q_transform.untransform(mu)
    # mu = (B*w).sum(axis=2)

    alpha = 0.05
    q_mean = mu.mean(axis=1)
    q_u = np.quantile(mu, 1-alpha/2, axis=1) 
    q_l = np.quantile(mu, alpha/2, axis=1)
    ax.plot(q_mean, h, color='black')
    ax.fill_betweenx(h, x1=q_u, x2=q_l, color='lightgray')
