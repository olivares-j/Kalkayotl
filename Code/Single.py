import sys
import numpy as np
import pymc3 as pm


class Single1D(pm.Model):
    '''
    Model to infer the distance of a series of stars
    '''
    def __init__(self,mu_data=None,Sigma_data=None,
        prior={"type":"Gaussian","location":300,"scale":20},
        name='single_1d', model=None):
        # 2) call super's init first, passing model and name
        # to it name will be prefix for all variables here if
        # no name specified for model there will be no prefix
        super().__init__(name, model)
        # now you are in the context of instance,
        # `modelcontext` will return self you can define
        # variables in several ways note, that all variables
        # will get model's name prefix

        #------------------- Data ------------------------------------------------------
        self.N = len(mu_data)

        if self.N == 0:
            sys.exit("Data has length zero!. You must provide at least one data point")

        self.T = np.linalg.inv(Sigma_data)
        #-------------------------------------------------------------------------------

        #------------------- Priors -------------------------------------------
        if prior["type"] == "Gaussian":
            self.Var('dist', Normal.dist(mu=prior["location"], sigma=theta[1],shape=self.N))

        elif prior["type"] == "Cauchy":
            self.Var('dist', Cauchy.dist(alpha=prior["location"], beta=theta[1],shape=self.N))

        elif prior["type"] == "Uniform":
            pm.Uniform('dist',lower=prior["location"]-prior["scale"],
                                        upper=prior["location"]+prior["scale"],shape=self.N)

        elif prior["type"] == "GMM":
            self.Var('dist', NormalMixture.dist(w=prior["weight"],mu=prior["location"],sigma=prior["scale"],comp_shape=1,shape=self.N))
        
        else:
            sys.exit("The specified prior is not supported")

        #----------------------- Transformation---------------------------------------
        pm.Deterministic('true_plx', 1/self.dist)
        #----------------------------------------------------------------------------

        #----------------------- Likelihood ----------------------------------------
        pm.MvNormal('plx_obs', mu=self.true_plx, tau=self.T,observed=mu_data)
        #------------------------------------------------------------------------------