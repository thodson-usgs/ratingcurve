import pymc3 as pm
from pymc3 import Model

import numpy as np

import patsy


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