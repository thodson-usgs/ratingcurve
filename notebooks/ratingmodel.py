import pymc as pm
from pymc import Model

import numpy as np

import patsy #just dmatrix

import aesara.tensor as at
#import theano.tensor as at
from ratingcurve.transforms import LogZTransform


class CustomModel(Model):
    # 1) override init
    def __init__(self, mean=0, sd=1, name='', model=None):
        super().__init__(name, model)
        pm.Normal('v2', mu=mean, sigma=sd)
        pm.Normal('v4', mu=mean, sigma=sd)
        
        
class RatingModel(Model):
    def __init__(self, name='', model=None):
        super().__init__(name, model)
    
    def setup(self, likelihood, prior):
        pass
    
    def fit(self):
        with model:
            mean_field = pm.fit(method="advi")
        # or pm.fit(method, model=self.model)
    
    def sample(self, n_samples, n_tune):
        with model:
            trace = pm.sample(50000)
  

class SplineRatingModel(RatingModel):
    ''' transform y, and compute D untransformed
    '''
    #def __init__(self, log_q, design, knots=5, mean=0, sd=1, name='', model=None):
    def __init__(self, q, dmatrix, knots, mean=0, sd=1, name='', model=None):
        super().__init__(name, model)
        
        # transform q
        self.q_obs = q
        self.q_transform = LogZTransform(q)
        y = self.q_transform.transform(self.q_obs)
        
        self.knots = knots
        self.B = dmatrix
        knot_dims = np.arange(self.B.shape[1])
        
        COORDS = {"obs" : np.arange(len(y)), "splines": np.arange(self.B.shape[1])}
        self.add_coords(COORDS)
        #self.add_coord("splines", values=np.arange(self.B.shape[1]))
        
        #a = pm.Normal("a", 0 , 1)
        w = pm.Normal("w", mu=mean, sd=sd, dims="splines")
        #mu = pm.Deterministic("mu", a + pm.math.dot(np.asarray(self.B, order="F"), w.T))
        mu = pm.Deterministic("mu", pm.math.dot(np.asarray(self.B, order="F"), w.T))
        #sigma = pm.Exponential("sigma", 1)
        sigma = pm.HalfCauchy("sigma", 1)
        D = pm.Normal("D", mu, sigma, observed=y, dims="obs")
       
    
class SegmentRatingModelOLD(RatingModel):
    ''' transform y
        assume strong priors on breaks
    '''
    #def __init__(self, log_q, design, knots=5, mean=0, sd=1, name='', model=None):
    def __init__(self,
                 q,
                 h,
                 breaks,
                 breaks_sigma,
                 name='',
                 model=None):
        super().__init__(name, model)
        
        # transform q
        self.q_obs = q
        self.q_transform = LogZTransform(self.q_obs)
        y = self.q_transform.transform(self.q_obs)
        
        # transform h (CHANGE TRANSFORM)
        self.h_obs = h
        bps = np.append(breaks, 200.0).reshape(-1,1)#.reshape(1,-1)
        bps_sigma = np.append(breaks_sigma, 0.1).reshape(-1,1)#.reshape(1,-1)
        
        h0_offset = np.ones_like(bps)
        h0_offset[0] = 0
        
        #self.h_transform = LogZTransform(self.h)
        #x = self.h_transform.transform(self.h_obs)
        
        
        
        COORDS = {"obs" : np.arange(len(y)), "splines": np.arange(len(breaks))}
        self.add_coords(COORDS)
        
        #priors
        
        #spline weights are positive; should we preference 0 or 1?
        #this is the power term so we have good prior based on physics
        #alternative a and w could be multinormal as in Reitan
        w = pm.HalfCauchy("w", beta=5, dims="splines") #beta=1
        a = pm.Normal("a", mu=0, sigma=10) #sigma=1
        #w = pm.Normal("w", mu=2.08, sigma=0.3, dims="splines")
        #import pdb; pdb.set_trace()
        #hs = pm.Normal('hs', mu=self.breaks, sigma=self.breaks_sigma)
        #import pdb; pdb.set_trace()
        hs = pm.Normal('hs', mu=bps, sigma=bps_sigma, shape=bps.shape)
        
        #h_s_prior = pm.Normal('h_s', mu=1, sd=1, shape=segments)
        h0 = hs - h0_offset
        
        #hs = pm.Normal("hs"
        #h0 = pm.Deterministic("h0", 
        
        # XXX revise if we transform h
        bo = at.switch( at.le(h, hs[:,:-1]), 0.0, at.log(h-h0))
        b1 = at.switch( at.gt(h, hs[:,1:]), bo, at.log(hs[:,1:]-h0))
        #bo = at.switch( at.le(h, hs[:,:-1]), 0.0, at.log(h-h0))
        #b1 = at.switch( at.le(h, hs[:,1:]), bo, at.log(hs[:,1:]-h0))
        #bo = at.switch( at.le(h, hs[:,:-1]), 0, at.log(h-h0[:,1:]))
        #b1 = at.switch( at.le(h, hs[:,1:]), bo, at.log(hs[:,1:]-h0[:,1:]))
        B = aesara.function([h], b1)
        
        
        mu = pm.Deterministic("mu", a + at.dot(B(h), w.T))
        #mu = pm.Deterministic("mu", pm.math.dot(np.asarray(self.B, order="F"), w.T))
        #sigma = pm.Exponential("sigma", 1)
        
        sigma = pm.HalfCauchy("sigma", 1)
        D = pm.Normal("D", mu, sigma, observed=y, dims="obs")
        
class SegmentRatingModel(RatingModel):
    ''' transform y
        assume strong priors on breaks
    '''
    #def __init__(self, log_q, design, knots=5, mean=0, sd=1, name='', model=None):
    def __init__(self,
                 q,
                 h,
                 breaks,
                 breaks_sigma,
                 name='',
                 model=None):
        super().__init__(name, model)
        
        # transform q
        self.q_obs = q
        self.q_transform = LogZTransform(self.q_obs)
        y = self.q_transform.transform(self.q_obs)
        
        # transform h (CHANGE TRANSFORM)
        self.h_obs = h
        #bps = np.append(breaks, 200.0).reshape(1,-1)
        #bps_sigma = np.append(breaks_sigma, 0.1).reshape(1,-1)
        
        h0_offset = np.ones_like(breaks)
        h0_offset[0] = 0
        
        #self.h_transform = LogZTransform(self.h)
        #x = self.h_transform.transform(self.h_obs)
        
        
        
        COORDS = {"obs" : np.arange(len(y)), "splines": np.arange(len(breaks))}
        self.add_coords(COORDS)
        
        #priors
        hs = pm.Normal("hs", mu=breaks, sigma=breaks_sigma, shape=(3,))
        #spline weights are positive; should we preference 0 or 1?
        #this is the power term so we have good prior based on physics
        #alternative a and w could be multinormal as in Reitan
        w = pm.HalfCauchy("w", beta=5, dims="splines") #beta=1
        a = pm.Normal("a", mu=0, sigma=10) #sigma=1
        
        #h_s_prior = pm.Normal('h_s', mu=1, sd=1, shape=segments)
        h0 = hs - h0_offset
        
        #indexing functions
        #test = at.switch( at.
        
       
        # XXX revise if we transform h 
        bo = at.switch( at.le(h, hs[:,:-1]), 0, at.log(h-h0))
        b1 = at.switch( at.le(h, hs[:,1:]), bo, at.log(hs[:,1:]-h0))
        B = aesara.function([h], b1)
        
        
        mu = pm.Deterministic("mu", a + at.dot(B(h), w.T))
        #mu = pm.Deterministic("mu", pm.math.dot(np.asarray(self.B, order="F"), w.T))
        #sigma = pm.Exponential("sigma", 1)
        
        sigma = pm.HalfCauchy("sigma", 1)
        D = pm.Normal("D", mu, sigma, observed=y, dims="obs")
        
