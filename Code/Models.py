import sys
import numpy as np
import pymc3 as pm
import theano
from theano import tensor as tt, printing

from Transformations import Iden,pc2mas,cartesianToSpherical,phaseSpaceToAstrometry,phaseSpaceToAstrometry_and_RV

from EDSD import EDSD


##################################3 Model 1D ####################################
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
		hyper_delta=None,
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

		#------------------- Data ------------------------------------------------------
		self.N = len(mu_data)

		if self.N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")

		self.T = np.linalg.inv(sg_data)
		#-------------------------------------------------------------------------------

		#============= Transformations ====================================

		if transformation is "mas":
			Transformation = Iden

		elif transformation is "pc":
			Transformation = pc2mas

		else:
			sys.exit("Transformation is not accepted")
		#==================================================================

		#================ Hyper-parameters =====================================
		if hyper_delta is None:
			shape = 1
		else:
			shape = len(hyper_delta)

		#------------------------ Location ----------------------------------
		if parameters["location"] is None:
			pm.Uniform("loc",lower=hyper_alpha[0][0],
							upper=hyper_alpha[0][1],
							shape=shape)

		else:
			self.loc = parameters["location"]

		#------------------------ Scale ---------------------------------------
		if parameters["scale"] is None:
			pm.HalfCauchy("scl",beta=hyper_beta[0],shape=shape)
		else:
			self.scl = parameters["scale"]
		#========================================================================

		#================= True values ========================================================
		#--------- Cluster oriented prior-----------------------------------------------
		if prior is "Uniform":
			pm.Uniform("source",lower=self.loc-self.scl,
								upper=self.loc+self.scl,shape=self.N)

		elif prior is "Cauchy":
			pm.Cauchy("source",alpha=self.loc,beta=self.scl,shape=self.N)

		elif prior is "Gaussian":
			pm.Normal("source",mu=self.loc,sd=self.scl,shape=self.N)

		elif prior is "GMM":
			pm.Dirichlet("weights",a=hyper_delta)

			pm.NormalMixture("source",w=self.weights,
				mu=self.loc,
				sigma=self.scl,
				comp_shape=1,
				shape=self.N)
		#---------- Galactic oriented prior ---------------------------------------------
		elif prior is "Half-Cauchy":
			pm.HalfCauchy("source",beta=self.scl,shape=self.N)

		elif prior is "Half-Gaussian":
			pm.HalfNormal("source",sigma=self.scl,shape=self.N)

		elif prior is "EDSD":
			EDSD("source",scale=self.scl,shape=self.N)
		
		else:
			sys.exit("The specified prior is not supported")
		#-----------------------------------------------------------------------------
		#=======================================================================================

		#----------------- Transformations ----------------------
		true = Transformation(self.source)

		#----------------------- Likelihood ----------------------------------------
		pm.MvNormal('obs', mu=true, tau=self.T,observed=mu_data)
		#------------------------------------------------------------------------------
####################################################################################################

############################ ND Model ###########################################################
class ModelND(pm.Model):
	'''
	Model to infer the N-dimensional parameter vector of a cluster
	'''
	def __init__(self,dimension,mu_data,sg_data,
		prior="Gaussian",
		parameters={"location":None,"scale":None,"corr":False},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		transformation=None,
		name='flavour_', model=None):

		assert isinstance(dimension,int), "dimension must be integer!"
		assert dimension in [3,5,6],"Not a valid dimension!"

		D = dimension

		# 2) call super's init first, passing model and name
		# to it name will be prefix for all variables here if
		# no name specified for model there will be no prefix
		super().__init__(name+str(D)+"d", model)
		# now you are in the context of instance,
		# `modelcontext` will return self you can define
		# variables in several ways note, that all variables
		# will get model's name prefix


		#------------------- Data ------------------------------------------------------
		N = int(len(mu_data)/D)


		if N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")

		T = np.linalg.inv(sg_data)
		#-------------------------------------------------------------------------------

		#============= Transformations ====================================

		if transformation is "mas":
			Transformation = Iden

		elif transformation is "pc":
			if D is 3:
				Transformation = cartesianToSpherical
			elif D is 6:
				Transformation = phaseSpaceToAstrometry_and_RV
			elif D is 5:
				Transformation = phaseSpaceToAstrometry
				D = 6

		else:
			sys.exit("Transformation is not accepted")
		#==================================================================

		#================ Hyper-parameters =====================================
		if hyper_delta is None:
			shape = 1
		else:
			shape = len(hyper_delta)

		#--------- Location ----------------------------------
		if parameters["location"] is None:

			location = [ pm.Uniform("loc_{0}".format(i),
						lower=hyper_alpha[i][0],
						upper=hyper_alpha[i][1],
						shape=shape) for i in range(D) ]

			#--------- Join variables --------------
			mu = pm.math.stack(location,axis=1)

		else:
			mu = parameters["location"]
		#------------------------------------------------------

		#------------- Scale --------------------------
		if parameters["scale"] is None:

			scale = [ pm.HalfCauchy("scl_{0}".format(i),
						beta=hyper_beta[i],
						shape=shape) for i in range(D) ]

		else:
			scale = parameters["scale"]
		#--------------------------------------------------

		#----------------------- Correlation -----------------------------------------
		if parameters["corr"] :
			pm.LKJCorr('chol_corr', eta=hyper_gamma, n=D)
			C = tt.fill_diagonal(self.chol_corr[np.zeros((D, D),dtype=np.int64)], 1.)
		else:
			C = np.eye(D)
		#-----------------------------------------------------------------------------

		#-------------------- Covariance -------------------------
		sigma_diag  = pm.math.stack(scale,axis=1)
		cov         = theano.shared(np.zeros((shape,D,D)))

		for i in range(shape):
			sigma       = tt.nlinalg.diag(sigma_diag[i])
			covi        = tt.nlinalg.matrix_dot(sigma, C, sigma)
			cov         = tt.set_subtensor(cov[i],covi)
		#---------------------------------------------------------
		#========================================================================

		#===================== True values ============================================
		if prior is "Gaussian":
			pm.MvNormal("source",mu=mu,cov=cov[0],shape=(N,D))

		elif prior is "GMM":
			pm.Dirichlet("weights",a=hyper_delta,shape=shape)

			comps = [ pm.MvNormal.dist(mu=mu[i],cov=cov[i]) for i in range(shape)] 

			pm.Mixture("source",w=self.weights,comp_dists=comps,shape=(N,D))
		
		else:
			sys.exit("The specified prior is not supported")
		#=================================================================================

		#----------------------- Transformation---------------------------------------
		transformed = Transformation(self.source)
		#-----------------------------------------------------------------------------

		#------------ Flatten --------------------------------------------------------
		true = pm.math.flatten(transformed)
		#----------------------------------------------------------------------------

		#----------------------- Likelihood ----------------------------------------
		pm.MvNormal('obs', mu=true, tau=T,observed=mu_data)
		#------------------------------------------------------------------------------