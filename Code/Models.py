import sys
import numpy as np
import pymc3 as pm


class Model1D(pm.Model):
    '''
    Model to infer the distance of a series of stars
    '''
    def __init__(self,mu_data=None,Sigma_data=None,
        prior="Gaussian",
        hyper_parameters={"location":1000,"scale":100},
        name='flavour_1d', model=None):
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
        if prior == "Gaussian":
            pm.HalfCauchy("location",beta=hyper_parameters["location"])
            pm.HalfCauchy("scale",beta=hyper_parameters["scale"])
            pm.Normal("distance",mu=self.location, sd=self.scale, shape=self.N)

        elif prior == "Cauchy":
            pm.HalfCauchy("location",beta=hyper_parameters["location"])
            pm.HalfCauchy("scale",beta=hyper_parameters["scale"])
            pm.Cauchy("distance",alpha=self.location, beta=self.scale, shape=self.N)

        elif prior == "Uniform":
            pm.HalfCauchy("location",beta=hyper_parameters["location"])
            pm.HalfCauchy("scale",beta=hyper_parameters["scale"])
            pm.Uniform("distance",lower=self.location - self.scale,
                              upper=self.location + self.scale,
                              shape=self.N)

        elif prior == "GMM":
            pm.HalfCauchy("location",beta=hyper_parameters["location"],
                                    shape=len(hyper_parameters["alpha"]))
            pm.HalfCauchy("scale",beta=hyper_parameters["scale"],
                                    shape=len(hyper_parameters["alpha"]))
            pm.Dirichlet("weights",a=hyper_parameters["alpha"],
                                    shape=len(hyper_parameters["alpha"]))

            pm.NormalMixture("distance",w=self.weights,mu=self.location,sigma=self.scale,
                comp_shape=1,shape=self.N)
        
        else:
            sys.exit("The specified prior is not supported")

        #----------------------- Transformation---------------------------------------
        pm.Deterministic('true_plx', 1/self.distance)
        #----------------------------------------------------------------------------

        #----------------------- Likelihood ----------------------------------------
        pm.MvNormal('plx_obs', mu=self.true_plx, tau=self.T,observed=mu_data)
        #------------------------------------------------------------------------------