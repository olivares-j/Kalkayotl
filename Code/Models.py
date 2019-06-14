import sys
import numpy as np
import pymc3 as pm

#-------- Import Theano transformations ------------


class Model1D(pm.Model):
    '''
    Model to infer the distance of a series of stars
    '''
    def __init__(self,mu_data=None,Sigma_data=None,
        prior="Gaussian",
        parameters={"location":None,"scale": None},
        hyper_alpha=[[0,10]],
        hyper_beta=[0.5],
        hyper_gamma=None,
        transformation=None,
        name='flavour_1d', model=None):
        # 2) call super's init first, passing model and name
        # to it name will be prefix for all variables here if
        # no name specified for model there will be no prefix
        super().__init__(name, model)
        # now you are in the context of instance,
        # `modelcontext` will return self you can define
        # variables in several ways note, that all variables
        # will get model's name prefix

        self.lower_bound_sd = 1e-5

        #------------------- Data ------------------------------------------------------
        self.N = len(mu_data)

        if self.N == 0:
            sys.exit("Data has length zero!. You must provide at least one data point")

        self.T = np.linalg.inv(Sigma_data)
        #-------------------------------------------------------------------------------

        #============= Transformations ====================================
        # if transformation is None:
        #     self.Transformation = Identity
        # elif transformation is "GAL2EQ":
        #     self.Transformation = GAL2EQ
        # else:
        #     sys.exit("Transformation is not accepted")
        #==================================================================
        #================ Hyper-parameters =====================================
        if hyper_gamma is None:
            shape = 1
        else:
            shape = len(hyper_gamma)

        #------------------------ Location ----------------------------------
        if parameters["location"] is None:
            pm.Uniform("mu",lower=hyper_alpha[0][0],
                            upper=hyper_alpha[0][1],
                            shape=shape)

        else:
            self.mu = parameters["location"]

        #------------------------ Scale ---------------------------------------
        if parameters["scale"] is None:
            pm.HalfCauchy("sd",beta=hyper_beta[0],shape=shape)
        else:
            self.sd = parameters["scale"]
        #========================================================================

        #------------------ True values ---------------------------------------------
        if prior is "Gaussian":
            pm.Normal("source",mu=self.mu,sd=self.sd,shape=self.N)

        elif prior is "GMM":
            pm.Dirichlet("weights",a=hyper_gamma,shape=shape)

            pm.NormalMixture("distances",w=self.weights,
                mu=self.mu,
                sigma=self.sd,
                comp_shape=1,
                shape=self.N)
        
        else:
            sys.exit("The specified prior is not supported")
        #-----------------------------------------------------------------------------

        #----------------------- Transformation---------------------------------------
        # pm.Deterministic('true', self.Transformation(self.source))
        pm.Deterministic('true', 1/self.source)
        #----------------------------------------------------------------------------

        #----------------------- Likelihood ----------------------------------------
        pm.MvNormal('obs', mu=self.true, tau=self.T,observed=mu_data)
        #------------------------------------------------------------------------------

        


class Model3D(pm.Model):
    '''
    Model to infer the distance and ra dec position of a cluster
    '''
    def __init__(self,mu_data=None,Sigma_data=None,
        prior="Gaussian",
        parameters={"location":None,"scale":None},
        hyper_alpha=[[0,360],[-90,90],[0,10]],
        hyper_beta=[10,10,0.5],
        hyper_gamma=None,
        transformation=None,
        name='flavour_3d', model=None):
        # 2) call super's init first, passing model and name
        # to it name will be prefix for all variables here if
        # no name specified for model there will be no prefix
        super().__init__(name, model)
        # now you are in the context of instance,
        # `modelcontext` will return self you can define
        # variables in several ways note, that all variables
        # will get model's name prefix

        #------------------- Data ------------------------------------------------------
        N = int(len(mu_data)/3)


        if N == 0:
            sys.exit("Data has length zero!. You must provide at least one data point")

        T = np.linalg.inv(Sigma_data)
        #-------------------------------------------------------------------------------

        #============= Transformations ====================================
        # if transformation is None:
        #     self.Transformation = Identity
        # elif transformation is "GAL2EQ":
        #     self.Transformation = GAL2EQ
        # else:
        #     sys.exit("Transformation is not accepted")
        #==================================================================
        #================ Hyper-parameters =====================================
        if hyper_gamma is None:
            shape = 1
        else:
            shape = len(hyper_gamma)

        #------------------------ Location ----------------------------------
        if parameters["location"] is None:
            pm.Uniform("location_0",lower=hyper_alpha[0][0],
                                    upper=hyper_alpha[0][1],
                                    shape=shape)
            pm.Uniform("location_1",lower=hyper_alpha[1][0],
                                    upper=hyper_alpha[1][1],
                                    shape=shape)
            pm.Uniform("location_2",lower=hyper_alpha[2][0],
                                    upper=hyper_alpha[2][1],
                                    shape=shape)

            #--------- Join variables --------------
            self.mu = pm.math.concatenate([self.location_0,
                                 self.location_1,
                                 self.location_2],axis=0)

        else:
            self.mu = np.array(parameters["location"])

        #------------------------ Scale ---------------------------------------
        if parameters["scale"] is None:
            packed_chol = pm.LKJCholeskyCov('chol_cov', eta=hyper_beta[1], n=3, sd_dist=pm.HalfCauchy.dist(beta=hyper_beta[0]))
            chol = pm.expand_packed_triangular(3, packed_chol, lower=True)

        else:
            chol = np.linalg.cholesky(parameters["scale"])

        
        #========================================================================

        

        #------------------ True values ---------------------------------------------
        if prior is "Gaussian":
            pm.MvNormal("source",mu=self.mu,chol=chol,shape=(N,3))

        elif prior is "GMM":
            pm.Dirichlet("weights",a=hyper_gamma,shape=shape)

            pm.NormalMixture("distances",w=self.weights,
                mu=self.mu,
                sigma=self.sd,
                comp_shape=1,
                shape=N)
        
        else:
            sys.exit("The specified prior is not supported")
        #-----------------------------------------------------------------------------


        #----------------------- Transformation---------------------------------------
        # pm.Deterministic('true', self.Transformation(self.source))
        self.source[:,2] = self.source[:,2]*1e-3
        pm.Deterministic('true', pm.math.flatten(self.source))

        #----------------------------------------------------------------------------

        #----------------------- Likelihood ----------------------------------------
        pm.MvNormal('obs', mu=self.true, tau=T,observed=mu_data)
        #------------------------------------------------------------------------------