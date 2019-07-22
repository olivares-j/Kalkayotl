import sys
import numpy as np
import pymc3 as pm
from theano import tensor as tt, printing

#-------- Import transformations ------------
def I1D(x):
    return x

def T1(x):
    return 1.e3/x
#---------------------------------------------


class Model1D(pm.Model):
    '''
    Model to infer the distance of a series of stars
    '''
    def __init__(self,mu_data,sg_data,
        prior="Gaussian",
        parameters={"location":None,"scale": None},
        hyper_alpha=[[0,10]],
        hyper_beta=[0.5],
        hyper_gamma=None,
        transformation="mas",
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

        self.T = np.linalg.inv(sg_data)
        #-------------------------------------------------------------------------------

        #============= Transformations ====================================

        if transformation is "mas":
            Transformation = I1D

        elif transformation is "pc":
            Transformation = T1

        else:
            sys.exit("Transformation is not accepted")
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
            pm.Dirichlet("weights",a=hyper_gamma)

            pm.NormalMixture("source",w=self.weights,
                mu=self.mu,
                sigma=self.sd,
                comp_shape=1,
                shape=self.N)
        
        else:
            sys.exit("The specified prior is not supported")
        #-----------------------------------------------------------------------------

        #----------------- Transformations ----------------------
        pm.Deterministic('true', Transformation(self.source))

        #----------------------- Likelihood ----------------------------------------
        pm.MvNormal('obs', mu=self.true, tau=self.T,observed=mu_data)
        #------------------------------------------------------------------------------

        
def I3D(x):
    return x

def cartesianToSpherical(a):
    """
    Convert Cartesian to spherical coordinates. The input can be scalars or 1-dimensional numpy arrays.
    Note that the angle coordinates follow the astronomical convention of using elevation (declination,
    latitude) rather than its complement (pi/2-elevation), which is commonly used in the mathematical
    treatment of spherical coordinates.
    Parameters
    ----------

    x - Cartesian vector component along the X-axis
    y - Cartesian vector component along the Y-axis
    z - Cartesian vector component along the Z-axis
    Returns
    -------

    The spherical coordinates r=sqrt(x*x+y*y+z*z), longitude phi, latitude theta.

    NOTE THAT THE LONGITUDE ANGLE IS BETWEEN 0 AND +2PI. FOR r=0 AN EXCEPTION IS RAISED.
    """
    x = a[:,0]
    y = a[:,1]
    z = a[:,2]
    rCylSq=x*x+y*y
    r=tt.sqrt(rCylSq+z*z)
    # if np.any(r==0.0):
    #   raise Exception("Error: one or more of the points is at distance zero.")
    phi = tt.arctan2(y,x)
    phi = tt.where(phi<0.0, phi+2*np.pi, phi)
    theta = tt.arctan2(z,tt.sqrt(rCylSq))
    #-------- Units----------
    phi   = tt.rad2deg(phi)   # Degrees
    theta = tt.rad2deg(theta) # Degrees
    plx   = 1000.0/r           # mas
    #------- Join ------
    res = tt.stack([phi, theta ,plx],axis=1)
    return res


class Model3D(pm.Model):
    '''
    Model to infer the distance and ra dec position of a cluster
    '''
    def __init__(self,mu_data,sg_data,
        prior="Gaussian",
        parameters={"location":None,"scale":None},
        hyper_alpha=None,
        hyper_beta=None,
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

        T = np.linalg.inv(sg_data)
        #-------------------------------------------------------------------------------

        #============= Transformations ====================================

        if transformation is "mas":
            Transformation = I3D

        elif transformation is "pc":
            Transformation = cartesianToSpherical

        else:
            sys.exit("Transformation is not accepted")
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
            mu = pm.math.concatenate([self.location_0,
                                 self.location_1,
                                 self.location_2],axis=0)

        else:
            mu = parameters["location"]

        #------------------------ Scale ---------------------------------------
        if parameters["scale"] is None:
            pm.HalfCauchy("sd_0",beta=hyper_beta[0],shape=shape)
            pm.HalfCauchy("sd_1",beta=hyper_beta[1],shape=shape)
            pm.HalfCauchy("sd_2",beta=hyper_beta[2],shape=shape)

            sigma_diag  = pm.math.concatenate([self.sd_0,self.sd_1,self.sd_2],axis=0)
            sigma       = tt.nlinalg.diag(sigma_diag)


            # pm.LKJCorr('chol_corr', eta=hyper_beta[3], n=3)
            # pm.Deterministic('C', tt.fill_diagonal(self.chol_corr[np.zeros((3, 3), dtype=np.int64)], 1.))
            self.C = np.eye(3)

            cov= tt.nlinalg.matrix_dot(sigma, self.C, sigma)
           

        else:
            cov = parameters["scale"]
        #========================================================================

        #------------------ True values ---------------------------------------------
        if prior is "Gaussian":
            pm.MvNormal("source",mu=mu,cov=cov,shape=(N,3))

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
        pm.Deterministic('transformed', Transformation(self.source))
        # transformed_print = tt.printing.Print("transformed")(self.transformed)
        #-----------------------------------------------------------------------------

        #------------ Flatten --------------------------------------------------------
        pm.Deterministic('true', pm.math.flatten(self.transformed))
        #----------------------------------------------------------------------------

        #----------------------- Likelihood ----------------------------------------
        pm.MvNormal('obs', mu=self.true, tau=T,observed=mu_data)
        #------------------------------------------------------------------------------