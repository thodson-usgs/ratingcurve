import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
    #q_obs = model.q_transform.transform(q_obs)
    
    a = trace.posterior['a'].mean(dim=['chain','draw']).data
    w = trace.posterior['w'].mean(dim=['chain','draw']).data
    hs = trace.posterior['hs'].mean(dim=['chain','draw']).data
    
    _plot_transitions(hs, 0, q_obs.max(), ax=axes)
    
    _plot_gagings(h_obs, q_obs, ax=axes)
    
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
    q_obs = model.q_obs
    h_obs = model.h_obs
    #q_obs = model.q_transform.transform(q_obs)
    
    a = trace.posterior['a'].mean(dim=['chain','draw']).data
    w = trace.posterior['w'].mean(dim=['chain','draw']).data
    hs = trace.posterior['hs'].mean(dim=['chain','draw']).data
    
    _plot_transitions(hs, 0, q_obs.max(), ax=axes)
    
    _plot_gagings(h_obs, q_obs, ax=axes)
    
    _plot_rating2(a, w, hs,
                 h_min=h_obs.min(),
                 h_max=h_obs.max(),
                 transform=model.q_transform,
                 ax=axes)
    
    # label
    axes.set_ylabel('Stage')
    axes.set_xlabel('Discharge')
    
    
def _plot_transitions(hs, q_min, q_max , ax=None):
    ax.hlines(hs, q_min, q_max, color='grey', linestyle='dotted')
        
    
def _plot_gagings(h_obs, q_obs, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5,5))
        
    ax.scatter(y=h_obs, x=q_obs, marker='o')

def _plot_rating(a, w, hs, h_min, h_max, transform=None, ax=None):
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
 