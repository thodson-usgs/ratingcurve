import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import aesara.tensor as at


def plot_power_law_rating(model, trace, colors = ('tab:blue', 'tab:orange'), ax=None):
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
        fig, axes = plt.subplots(1, figsize(5,5))
    
    q_obs = model.q_obs.flatten()
    h_obs = model.h_obs.flatten()
    if model.q_sigma is not None:
        q_sigma = model.q_sigma.flatten()
    else:
        q_sigma = None
    #q_obs = model.q_transform.transform(q_obs)
    
    a = trace.posterior['a'].mean(dim=['chain','draw']).data
    w = trace.posterior['w'].mean(dim=['chain','draw']).data
    hs = trace.posterior['hs'].mean(dim=['chain','draw']).data
    
    _plot_transitions(trace.posterior['hs'],
                      0, q_obs.max(), ax=axes)
    
    _plot_gagings(h_obs, q_obs, q_sigma, ax=axes)
    
    _plot_powerlaw_rating(a, w, hs,
                          h_min=h_obs.min(),
                          h_max=h_obs.max(),
                          transform=model.q_transform,
                          ax=axes)
    
    # label
    axes.set_ylabel('Stage')
    axes.set_xlabel('Discharge')
    
    
def _plot_transitions(hs, q_min, q_max , ax=None):
    alpha = 0.05
    hs_u = hs.mean(dim=['chain','draw']).data
    hs_lower = hs.quantile(alpha/2, dim=['chain','draw']).data
    hs_upper = hs.quantile(1- alpha/2, dim=['chain','draw']).data
    
    [ax.axhspan(l, u, color='whitesmoke') for u,l in zip(hs_lower, hs_upper)]
    [ax.axhline(u, color='grey', linestyle='dotted') for u in hs_u]
    
def _plot_gagings(h_obs, q_obs, q_sigma=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5,5))
        
    #ax.scatter(y=h_obs, x=q_obs, marker='o')
    if q_sigma is not None:
        q_sigma = q_sigma * 1.96
        
    ax.errorbar(y=h_obs, x=q_obs, xerr=q_sigma, fmt="o")
        

def _plot_powerlaw_rating(a, w, hs, h_min, h_max, transform=None, ax=None):
    ''' TODO REvise to numpy with np.where
    TODO or create function that is reused in rating model
    '''
    
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5,5))
    
    inf = at.constant([np.inf])
    
    clips_array = np.zeros(len(hs))
    clips_array[0] = -np.inf
    clips = at.constant(clips_array)
    # TODO log distribute
    h = np.linspace(h_min, h_max,100).reshape(-1,1)
    
    hsi = at.concatenate([hs, inf])[1:]
    h0_offset = np.ones(len(hs))
    h0_offset[0] = 0
    
    h0 = hs - h0_offset
    
    b1 = at.switch( at.le(h, hs), clips , at.log(h-h0))
    mu = (a + at.dot(b1, w)).eval()
     
    if transform:
        mu = transform.untransform(mu)
        
    ax.plot(mu, h, color='black')