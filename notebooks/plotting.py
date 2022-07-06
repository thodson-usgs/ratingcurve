import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import theano.tensor as at
import aesara.tensor as at

def plot_power_law_rating(model, trace, colors = ('tab:blue', 'tab:orange')):
                          
                          
    """
    Plots segmented power law rating model
    ----------
    trace : trace returned by model
    h_obs :
    q_obs :
    colors : list with 2 colornames
    Returns
    -------
    figure, axes
    """
    # plot rating curve and error curve side-by-side                      
    #fig, axes = plt.subplots(2, 2, figsize=(9, 5),
    #                         gridspec_kw={'height_ratios': [1, 3],
    #                         'width_ratios': [2, 3]})
    fig, axes = plt.subplots(1, figsize=(5,5))
    
    #axes.scatter(y=h_obs, x=q_obs)
    q_obs = model.q_obs
    h_obs = model.h_obs
    q_sigma = model.q_sigma
    #q_obs = model.q_transform.transform(q_obs)
    
    #pymc4
    a = trace.posterior['a'].mean(dim=['chain','draw']).data
    w = trace.posterior['w'].mean(dim=['chain','draw']).data
    hs = trace.posterior['hs'].mean(dim=['chain','draw']).data
    #a = trace['a'].mean()
    #w = trace['w'].mean()
    #hs = trace['hs'].mean()
    
    _plot_transitions(hs, 0, q_obs.max(), ax=axes)
    
    _plot_gagings(h_obs, q_obs, q_sigma, ax=axes)
    
    _plot_rating(a, w, hs,
                 h_min=h_obs.min(),
                 h_max=h_obs.max(),
                 transform=model.q_transform,
                 ax=axes)
    
    # label
    axes.set_ylabel('Stage')
    axes.set_xlabel('Discharge')

def plot_power_law_rating2(model, trace, colors = ('tab:blue', 'tab:orange')):
                          
                          
    """
    Plots segmented power law rating model
    ----------
    trace : trace returned by model
    h_obs :
    q_obs :
    colors : list with 2 colornames
    Returns
    -------
    figure, axes
    """
    # plot rating curve and error curve side-by-side                      
    #fig, axes = plt.subplots(2, 2, figsize=(9, 5),
    #                         gridspec_kw={'height_ratios': [1, 3],
    #                         'width_ratios': [2, 3]})
    fig, axes = plt.subplots(1, figsize=(5,5))
    
    #axes.scatter(y=h_obs, x=q_obs)
    q_obs = model.q_obs.flatten()
    h_obs = model.h_obs.flatten()
    if model.q_sigma is not None:
        q_sigma = model.q_sigma.flatten()
    else:
        q_sigma = None
    #q_obs = model.q_transform.transform(q_obs)
    
    #pymc4
    a = trace.posterior['a'].mean(dim=['chain','draw']).data
    w = trace.posterior['w'].mean(dim=['chain','draw']).data
    hs = trace.posterior['hs'].mean(dim=['chain','draw']).data
    #a = np.array(trace['a'].mean())
    #w = np.array(trace['w'].mean())
    #hs = np.array(trace['hs'].mean())
    
    #pymc4
    _plot_transitions(trace.posterior['hs'],
    #_plot_transitions(trace['hs'],
                      0, q_obs.max(), ax=axes)
    
    _plot_gagings(h_obs, q_obs, q_sigma, ax=axes)
    
    _plot_rating2(a, w, hs,
                 h_min=h_obs.min(),
                 h_max=h_obs.max(),
                 transform=model.q_transform,
                 ax=axes)
    
    # label
    axes.set_ylabel('Stage')
    axes.set_xlabel('Discharge')
    
    
def _plot_transitions(hs, q_min, q_max , ax=None):
    alpha = 0.05
    #hs_u = hs.mean()
    #hs_lower = np.quantile(hs,alpha/2)
    #hs_upper = np.quantile(hs,1- alpha/2)
    #pymc4
    hs_u = hs.mean(dim=['chain','draw']).data
    hs_lower = hs.quantile(alpha/2, dim=['chain','draw']).data
    hs_upper = hs.quantile(1- alpha/2, dim=['chain','draw']).data
    
    #ax.hlines(hs_u, q_min, q_max, color='grey', linestyle='dotted')
    #if won't be necessary in pymc4 XXX
    #if type(hs_lower) == np.float64:
    #    ax.axhspan(hs_lower, hs_upper, color='whitesmoke')
    #    ax.axhline(hs_u, color='grey', linestyle='dotted')
    #    
    #else:
    #    [ax.axhspan(l, u, color='whitesmoke') for u,l in zip(hs_lower, hs_upper)]
    #    [ax.axhline(u, color='grey', linestyle='dotted') for u in hs_u]
    [ax.axhspan(l, u, color='whitesmoke') for u,l in zip(hs_lower, hs_upper)]
    [ax.axhline(u, color='grey', linestyle='dotted') for u in hs_u]
    #print(hs_u)
    #ax.hxvline(hs_u, color='grey', linestyle='dotted')
    #ax.axhspans(hs_lower, hs_upper)
    
def _plot_gagings(h_obs, q_obs, q_sigma=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5,5))
        
    #ax.scatter(y=h_obs, x=q_obs, marker='o')
    if q_sigma is not None:
        q_sigma = q_sigma * 1.96
        
    ax.errorbar(y=h_obs, x=q_obs, xerr=q_sigma, fmt="o")
        

def _plot_rating(a, w, hs, h_min, h_max, transform=None, ax=None):
    ''' TODO REvise to numpy with np.where
    TODO or create function that is reused in rating model
    '''
    
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5,5))
    
    inf = at.constant([np.inf])
    #if not needed in pymc4
    #if type(hs) == np.float64:
    #    clips_array = np.zeros(1)
    #else:
    #    clips_array = np.zeros(len(hs))
    clips_array = np.zeros(len(hs))
    clips_array[0] = -np.inf
    clips = at.constant(clips_array)
    # TODO log distribute
    h = np.linspace(h_min, h_max,100).reshape(-1,1)
    
    hsi = at.concatenate([hs, inf])[1:]
    h0_offset = np.ones(len(hs))
    h0_offset[0] = 0
    
    h0 = hs - h0_offset
    
    b0 = at.switch( at.le(h, hs), clips , at.log(h-h0))
    b1 = at.switch( at.le(h, hsi), b0, at.log(hsi-h0))
    #import pdb; pdb.set_trace()
    mu = (a + at.dot(b1, w)).eval()
     
    #q = np.exp(mu.eval()) #XXX
    if transform:
        mu = transform.untransform(mu)
        
    ax.plot(mu, h, color='black')
    #ax.plot(h, mu.eval(), color='black')
    #model
    
def _plot_rating2(a, w, hs, h_min, h_max, transform=None, ax=None):
    ''' TODO REvise to numpy with np.where
    TODO or create function that is reused in rating model
    '''
    
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5,5))
    
    inf = at.constant([np.inf])
    #if not needed in pymc4
    #if type(hs) == np.float64:
    #    clips_array = np.zeros(1)
    #else:
    #    clips_array = np.zeros(len(hs)) 
    
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
    #b1 = at.switch( at.le(h, hsi), b0, at.log(hsi-h0))
    #import pdb; pdb.set_trace()
    mu = (a + at.dot(b1, w)).eval()
     
    #q = np.exp(mu.eval()) #XXX
    if transform:
        mu = transform.untransform(mu)
        
    ax.plot(mu, h, color='black')
    #ax.plot(h, mu.eval(), color='black')
    #model
 