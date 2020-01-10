import sys
import numpy as np
import pymc3 as pm
import theano
from theano import tensor as tt, printing

from kalkayotl.Transformations import Iden,pc2mas,cartesianToSpherical,phaseSpaceToAstrometry,phaseSpaceToAstrometry_and_RV
from kalkayotl.Priors import EDSD,EFF,King

################################## Model 1D ####################################
class Model1D(pm.Model):
	'''
	Model to infer the distance of a series of stars
	'''
	def __init__(self,mu_data,tau_data,
		prior="Gaussian",
		parameters={"location":None,"scale": None},
		hyper_alpha=[100,10],
		hyper_beta=[10],
		hyper_gamma=None,
		hyper_delta=None,
		transformation="mas",
		parametrization="non-central",
		name='1D', model=None):
		super().__init__(name, model)

		#------------------- Data ------------------------------------------------------
		self.N = len(mu_data)

		if self.N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")

		#-------------------------------------------------------------------------------

		#============= Transformations ====================================

		if transformation is "mas":
			Transformation = Iden

		elif transformation is "pc":
			Transformation = pc2mas

		else:
			sys.exit("Transformation is not accepted")

		if parametrization == "non-central":
			print("Using non central parametrization.")
		else:
			print("Using central parametrization.")
		#==================================================================

		#================ Hyper-parameters =====================================
		if hyper_delta is None:
			shape = ()
		else:
			shape = len(hyper_delta)

		#------------------------ Location ----------------------------------
		if parameters["location"] is None:
			pm.Normal("loc",mu=hyper_alpha[0],sigma=hyper_alpha[1],shape=shape)

		else:
			self.loc = parameters["location"]

		#------------------------ Scale ---------------------------------------
		if parameters["scale"] is None:
			pm.Gamma("scl",alpha=2.0,beta=2.0/hyper_beta[0],shape=shape)
		else:
			self.scl = parameters["scale"]
		#========================================================================

		#================= True values ========================================================
		#--------- Cluster oriented prior-----------------------------------------------
		if prior is "Uniform":
			if parametrization == "central":
				pm.Uniform("source",lower=self.loc-self.scl,upper=self.loc+self.scl,shape=self.N)
			else:
				pm.Uniform("offset",lower=-1.,upper=1.,shape=self.N)
				pm.Deterministic("source",self.loc + self.scl*self.offset)

		elif prior is "Cauchy":
			if parametrization == "central":
				pm.Cauchy("source",alpha=self.loc,beta=self.scl,shape=self.N)
			else:
				pm.Cauchy("offset",alpha=0.0,beta=1.0,shape=self.N)
				pm.Deterministic("source",self.loc + self.scl*self.offset)

		elif prior is "Gaussian":
			if parametrization == "central":
				pm.Normal("source",mu=self.loc,sd=self.scl,shape=self.N)
			else:
				pm.Normal("offset",mu=0.0,sd=1.0,shape=self.N)
				pm.Deterministic("source",self.loc + self.scl*self.offset)

		elif prior is "GMM":
			if parameters["weights"] is None:
				pm.Dirichlet("weights",a=hyper_delta,shape=shape)
				# break symmetry
				pm.Potential('order_means', tt.switch(self.loc[1]-self.loc[0] < 0, -1e6, 0))
			else:
				self.weights = parameters["weights"]

			if parametrization == "central":
				pm.NormalMixture("source",w=self.weights,
					mu=self.loc,
					sigma=self.scl,
					comp_shape=1,
					shape=self.N)
			else:
				pm.Normal("offset",mu=0.0,sd=1.0,shape=self.N)
				# latent cluster of each observation
				component = pm.Categorical("component",p=self.weights,shape=self.N)
				pm.Deterministic("source",self.loc[component] + self.scl[component]*self.offset) 

		elif prior is "EFF":
			if parameters["gamma"] is None:
				pm.TruncatedNormal("gamma",mu=hyper_gamma[0],sigma=hyper_gamma[1],
									lower=2.0,upper=10.0)
			else:
				self.gamma = parameters["gamma"]

			if parametrization == "central":
				EFF("source",location=self.loc,scale=self.scl,gamma=self.gamma,shape=self.N)
			else:
				EFF("offset",location=0.0,scale=1.0,gamma=self.gamma,shape=self.N)
				pm.Deterministic("source",self.loc + self.scl*self.offset)

		elif prior is "King":
			if parameters["rt"] is None:
				pm.Gamma("x",alpha=2.0,beta=2.0/hyper_gamma[0])
				pm.Deterministic("rt",1.0+self.x)
			else:
				self.rt = parameters["rt"]

			if parametrization == "central":
				King("source",location=self.loc,scale=self.scl,rt=self.rt,shape=self.N)
			else:
				King("offset",location=0.0,scale=1.0,rt=self.rt,shape=self.N)
				pm.Deterministic("source",self.loc + self.scl*self.offset)
			
		#---------- Galactic oriented prior ---------------------------------------------
		elif prior == "Half-Cauchy":
			pm.HalfCauchy("source",beta=self.scl,shape=self.N)

		elif prior == "Half-Gaussian":
			pm.HalfNormal("source",sigma=self.scl,shape=self.N)

		elif prior is "EDSD":
			EDSD("source",scale=self.scl,shape=self.N)
		
		else:
			sys.exit("The specified prior is not implemented")
		#-----------------------------------------------------------------------------
		#=======================================================================================
		# print_ = tt.printing.Print("source")(self.source)
		#----------------- Transformations ----------------------
		true = Transformation(self.source)

		#----------------------- Likelihood ----------------------------------------
		pm.MvNormal('obs', mu=true, tau=tau_data,observed=mu_data)
		#------------------------------------------------------------------------------
####################################################################################################

############################ ND Model ###########################################################
class ModelND(pm.Model):
	'''
	Model to infer the N-dimensional parameter vector of a cluster
	'''
	def __init__(self,dimension,mu_data,tau_data,
		prior="Gaussian",
		parameters={"location":None,"scale":None,"corr":False},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		transformation=None,
		parametrization="non-central",
		name='', model=None):

		assert isinstance(dimension,int), "dimension must be integer!"
		assert dimension in [3,5,6],"Not a valid dimension!"

		D = dimension

		# 2) call super's init first, passing model and name
		# to it name will be prefix for all variables here if
		# no name specified for model there will be no prefix
		super().__init__(str(D)+"D", model)
		# now you are in the context of instance,
		# `modelcontext` will return self you can define
		# variables in several ways note, that all variables
		# will get model's name prefix


		#------------------- Data ------------------------------------------------------
		N = int(len(mu_data)/D)
		if N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")
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

			location = [ pm.Normal("loc_{0}".format(i),
						mu=hyper_alpha[i][0],
						sigma=hyper_alpha[i][1],
						shape=shape) for i in range(D) ]

			#--------- Join variables --------------
			mu = pm.math.stack(location,axis=1)

		else:
			mu = parameters["location"]
		#------------------------------------------------------

		#------------- Scale --------------------------
		if parameters["scale"] is None:
			scale = [ pm.Gamma("scl_{0}".format(i),
						alpha=2.0,beta=2.0/hyper_beta[i][0],
						shape=shape) for i in range(D) ]

		else:
			scale = parameters["scale"]
		#--------------------------------------------------

		#----------------------- Correlation -----------------------------------------
		if parameters["corr"] :
			pm.LKJCorr('chol_corr', eta=hyper_gamma, n=D)
			C = tt.fill_diagonal(self.chol_corr[np.zeros((D, D),dtype=np.int64)], 1.)
			# print_ = tt.printing.Print('C')(C)
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
		pm.MvNormal('obs', mu=true, tau=tau_data,observed=mu_data)
		#------------------------------------------------------------------------------