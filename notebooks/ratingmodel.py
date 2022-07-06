import pymc as pm
from pymc import Model

import numpy as np
import patsy #just dmatrix

#import theano.tensor as at
import aesara.tensor as at
#from rating.transforms import LogZTransform
from transforms import LogZTransform


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
        self.y = self.q_transform.transform(self.q_obs)
        
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
        D = pm.Normal("D", mu, sigma, observed=self.y, dims="obs")
       
    

class SegmentedRatingModel(RatingModel):
    ''' transform y
        assume uniform priors on breaks
        ALTERNATE PARAMETERIZATION
    '''
    #def __init__(self, log_q, design, knots=5, mean=0, sd=1, name='', model=None):
    def __init__(self,
                 q,
                 h,
                 segments,
                 prior={'distribution':'uniform'},
                 q_sigma=None,
                 name='',
                 model=None):
        
        super().__init__(name, model)
        
        self.segments = segments
        self.prior = prior 
        # transform q
        self.q_obs = q
        self.q_transform = LogZTransform(self.q_obs)
        self.y = self.q_transform.transform(self.q_obs)
        self.q_sigma = q_sigma
       
        ##XXX verify this is correct
        #if q_sigma is not None:
        #    self.y_sigma = self.q_sigma / self.q_obs.std()
        #else:
        #    self.y_sigma = 0
        # convert uncertainty to weights
        if q_sigma is not None:
            self.y_sigma = self.q_sigma / self.q_obs.std() #XXX 
            self._w = self.q_obs.var() / self.q_sigma**2
            self._w = self._w.flatten()
            
        else:
            self._w = 1
            self.y_sigma = 0
        
        self.h_obs = h
        
        self._inf = [np.inf]
        #self._inf = [np.array(99999.0)] #TESTING
        
        # clipping boundary
        clips = np.zeros(self.segments)
        #clips[0] = -np.inf
        clips[0] = -100
        self._clips = at.constant(clips)
        
        # create h0 offsets
        self._h0_offsets = np.ones(segments)
        self._h0_offsets[0] = 0
        
        
        self.COORDS = {"obs" : np.arange(len(self.y)), "splines":np.arange(segments)}
        
        #compute initval
        self._hs_lower_bounds = np.zeros(self.segments) + self.h_obs.min()
        self._hs_lower_bounds[0] = 0
        
        self._hs_upper_bounds = np.zeros(self.segments) + self.h_obs.max()
        self._hs_upper_bounds[0] = self.h_obs.min() - 1e-6 #XXX HACK
        
        # set random init on unit interval then scale based on bounds
        self._init_hs = np.random.rand(self.segments) \
                         * (self._hs_upper_bounds - self._hs_lower_bounds) \
                         + self._hs_lower_bounds
        
        self._init_hs = np.sort(self._init_hs) # not necessary?
        
        self.compile_model()
        #return self.compile_model()
       
    def set_normal_prior(self):
        '''
        prior={type='normal', mu=[], sigma=[]}
        '''
        with Model(coords=self.COORDS) as model:
            hs_ = pm.TruncatedNormal('hs_', 
                                 mu = self.prior['mu'],
                                 sigma = self.prior['sigma'],
                                 lower=self._hs_lower_bounds,
                                 upper=self._hs_upper_bounds,
                                 shape=self.segments,
                                 #testval=self._init_hs) # define a function to compute
                                 initval=self._init_hs) # define a function to compute
            
            hs = pm.Deterministic('hs', at.sort(hs_))
            
        return hs

    def set_uniform_prior(self):
        '''
        prior={distribution:'uniform'}
        '''
        with Model(coords=self.COORDS) as model:
            hs_ = pm.Uniform('hs_', 
                                 lower=self._hs_lower_bounds,
                                 upper=self._hs_upper_bounds,
                                 shape=self.segments,
                                 #testval=self._init_hs) # define a function to compute
                                 initval=self._init_hs) # define a function to compute
            
            hs = pm.Deterministic('hs', at.sort(hs_))
            
        return hs

    
    def set_beta_prior(self):
        '''
        XXX NOT WORKING YEST
        XXX MAKE SEGMENTS VARIABLE
        prior={distribution:'uniform'}
        '''
        with Model(coords=self.COORDS) as model:
            hs_ = pm.Beta('hs_',
                          alpha=np.array([0.5, 2.0, 1.0]),
                          beta=np.array([0.5, 1.0, 2.0]),
                          shape=self.segments,
                          initval=self._init_hs) # define a function to compute
                          #testval=self._init_hs) # define a function to compute
            
            scaled = at.sort(hs_) * (self._hs_upper_bounds - self._hs_lower_bounds) + self._hs_lower_bounds
            hs = pm.Deterministic('hs', scaled)
            
            
        return hs
 
    
    def compile_model(self):
        with Model(coords=self.COORDS) as model:
            
            w = pm.Normal("w", mu=0, sigma=3, dims="splines")
            a = pm.Normal("a", mu=0, sigma=5)
            
            #TODO move this logic into set_prior?
            if self.prior['distribution']=='normal':
                hs = self.set_normal_prior()
            
            elif self.prior['distribution']=='beta':
                hs = self.set_beta_prior()
                
            else:
                hs = self.set_uniform_prior()
           
            if self.segments > 1:
                h0 = hs - self._h0_offsets
                b = pm.Deterministic('b',
                                      at.switch( at.le(self.h_obs, hs), self._clips , at.log(self.h_obs-h0)) )
                
                mu = pm.Deterministic("mu", a + at.dot(b, w))
                
            # XXX: so far this else does nothing 
            else:
                b = pm.Deterministic('b', at.log(self.h_obs-hs))
                mu = pm.Deterministic("mu", a + at.dot(b, w))
                
            ## Can we use weighted regression instead?
            ## https://discourse.pymc.io/t/how-to-perform-weighted-inference/1825
            #if self.q_sigma is not None:
            #    sigma_obs = pm.Normal("sigma_obs", mu=0, sigma=self.y_sigma.flatten())
            #    mu += sigma_obs
                
            sigma = pm.HalfCauchy("sigma", beta=1) #initval=0.01
            #D = pm.Normal("D", mu, sigma, observed=self.y.flatten(), dims="obs")
            
            #if self.q_sigma is not None:
            #    w = 1 / self.y_sigma
            #w = 1
            #pymc4 version
            xxx = pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), self.y.flatten())
            #xxx = pm.Normal.dist(mu=mu, sigma=sigma).logp(self.y.flatten())
            y_ = pm.Potential('Y_obs', self._w * xxx)
            #D = pm.Normal("D", mu, sigma, observed=self.y.flatten(), dims="obs")