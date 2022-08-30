import pymc as pm
from pymc import Model

import numpy as np
import aesara.tensor as at
from patsy import dmatrix

from .transforms import LogZTransform
from .plotting import plot_power_law_rating, plot_spline_rating


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

    def fit(self, method="advi", n=150_000):
        mean_field = pm.fit(method=method, n=n, model=self.model)
        return mean_field

    def sample(self, n_samples, n_tune):
        with model:
            trace = pm.sample(50_000)


class Dmatrix():
    def __init__(self, knots, degree, form):
        self.form = f"{form}(stage, knots=knots, degree={degree}, include_intercept=True) - 1"
        self.knots = knots

    def transform(self, stage):
        return dmatrix(self.form, {"stage": stage, "knots": self.knots[1:-1]})


def compute_knots(minimum, maximum, n):
    ''' Return list of knots
    '''
    return np.linspace(minimum, maximum, n)


class SegmentedRatingModel(RatingModel):
    ''' transform y
        assume uniform priors on breaks
        ALTERNATE PARAMETERIZATION
    '''
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

        # #XXX verify this is correct
        # if q_sigma is not None:
        #     self.y_sigma = self.q_sigma / self.q_obs.std()
        # else:
        #     self.y_sigma = 0

        # convert uncertainty to weights
        if q_sigma is not None:
            self.y_sigma = self.q_sigma / self.q_obs.std() #XXX 

        else:
            self._w = 1
            self.y_sigma = 0

        self.h_obs = h

        self._inf = [np.inf]

        # clipping boundary
        clips = np.zeros(self.segments)
        #clips[0] = -np.inf
        clips[0] = -1000 #TODO verify whether this should be inf
        self._clips = at.constant(clips)

        # create h0 offsets
        self._h0_offsets = np.ones(segments)
        self._h0_offsets[0] = 0

        self.COORDS = {"obs" : np.arange(len(self.y)), "splines":np.arange(segments)}

        # compute initval
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

    def plot(self, trace, ax=None):
        plot_power_law_rating(self, trace, ax=ax)


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

    #def plot(self):
    #    plot_power_law_rating(self, trace, colors = ('tab:blue', 'tab:orange'), ax=None)

    def compile_model(self):
        with Model(coords=self.COORDS) as model:
            # TESING XXX
            h = pm.MutableData("h", self.h_obs)

            w = pm.Normal("w", mu=0, sigma=3, dims="splines")    
            a = pm.Normal("a", mu=0, sigma=5)

            #TODO move this logic into set_prior?
            if self.prior['distribution']=='normal':
                hs = self.set_normal_prior()

            elif self.prior['distribution']=='beta':
                hs = self.set_beta_prior()

            else:
                hs = self.set_uniform_prior()

            h0 = hs - self._h0_offsets # better convergence
            b = pm.Deterministic('b',
                                  at.switch( at.le(h, hs), self._clips , at.log(h-h0)) )

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
            y_ = pm.Potential('Y_obs', self._w * xxx)
            #D = pm.Normal("D", mu, sigma, observed=self.y.flatten(), dims="obs")


class SplineRatingModel(RatingModel):
    ''' transform y, and compute D untransformed
    '''

    def __init__(self, q, h, knots, mean=0, sd=1, name='', model=None):
        super().__init__(name, model)
        #TODO redefine priors
        # transform q
        self.q_obs = q
        self.h_obs = h
        self.q_transform = LogZTransform(self.q_obs)
        self.y = self.q_transform.transform(self.q_obs)
        #self.q_sigma = q_sigma
        self.knots = knots
        self._d_matrix = Dmatrix(knots, 3, 'bs')
        self.d_transform = self._d_matrix.transform #XXX rename?

        #self.B = pm.MutableData("B", self.d_transform(h))
        self.B = self.d_transform(h)
        B = pm.MutableData("B", self.B)
        knot_dims = np.arange(self.B.shape[1])

        COORDS = {"obs" : np.arange(len(self.y)), "splines": np.arange(self.B.shape[1])}
        self.add_coords(COORDS)
        #self.add_coord("splines", values=np.arange(self.B.shape[1]))

        #a = pm.Normal("a", 0 , 1)
        w = pm.Normal("w", mu=mean, sigma=sd, dims="splines")
        #mu = pm.Deterministic("mu", pm.math.dot(np.asarray(self.B, order="F"), w.T))
        mu = pm.Deterministic("mu", pm.math.dot(B, w.T))

        sigma = pm.HalfCauchy("sigma", 1)
        D = pm.Normal("D", mu, sigma, observed=self.y, dims="obs")

    def plot(self, trace, ax=None):
        plot_spline_rating(self, trace, ax=ax)