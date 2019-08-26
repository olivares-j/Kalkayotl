import sys
import numpy as np
import pymc3 as pm
import theano
from theano import tensor as tt, printing

from Transformations import Iden,pc2mas,cartesianToSpherical,cartesianToSpherical_plus_mu,phaseSpaceToAstrometry


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
			pm.Dirichlet("weights",a=hyper_delta)

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
####################################################################################################

######################################### 3D Model ################################################### 
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
		hyper_delta=None,
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
			Transformation = Iden

		elif transformation is "pc":
			Transformation = cartesianToSpherical

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
			mu = pm.math.stack([self.location_0,
								self.location_1,
								self.location_2],axis=1)

		else:
			mu = parameters["location"]

		#------------------------ Scale ---------------------------------------
		if parameters["scale"] is None:
			pm.HalfCauchy("sd_0",beta=hyper_beta[0],shape=shape)
			pm.HalfCauchy("sd_1",beta=hyper_beta[1],shape=shape)
			pm.HalfCauchy("sd_2",beta=hyper_beta[2],shape=shape)

			# pm.LKJCorr('chol_corr', eta=hyper_gamma, n=3)
			# pm.Deterministic('C', tt.fill_diagonal(self.chol_corr[np.zeros((3, 3), dtype=np.int64)], 1.))
			C = np.eye(3)

			sigma_diag  = pm.math.stack([self.sd_0,self.sd_1,self.sd_2],axis=1)

			cov = theano.shared(np.zeros((shape,3,3)))

			for i in range(shape):
				sigma       = tt.nlinalg.diag(sigma_diag[i])
				covi        = tt.nlinalg.matrix_dot(sigma, C, sigma)
				cov         = tt.set_subtensor(cov[i],covi)
			
			# cov_print   = tt.printing.Print("cov")(cov)
		   

		else:
			cov = parameters["scale"]
		#========================================================================

		#------------------ True values ---------------------------------------------
		if prior is "Gaussian":
			pm.MvNormal("source",mu=mu,cov=cov[0],shape=(N,3))

		elif prior is "GMM":
			pm.Dirichlet("weights",a=hyper_delta,shape=shape)

			comps = [ pm.MvNormal.dist(mu=mu[i],cov=cov[i]) for i in range(shape)] 

			pm.Mixture("source",w=self.weights,comp_dists=comps,shape=(N,3))
		
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

##################################################################################################

############################ 5D Model ###########################################################
class Model5D(pm.Model):
	'''
	Model to infer the position,distance and proper motions of a cluster
	'''
	def __init__(self,mu_data,sg_data,
		prior="Gaussian",
		parameters={"location":None,"scale":None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		transformation=None,
		name='flavour_5d', model=None):
		# 2) call super's init first, passing model and name
		# to it name will be prefix for all variables here if
		# no name specified for model there will be no prefix
		super().__init__(name, model)
		# now you are in the context of instance,
		# `modelcontext` will return self you can define
		# variables in several ways note, that all variables
		# will get model's name prefix

		#------------------- Data ------------------------------------------------------
		N = int(len(mu_data)/5)


		if N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")

		T = np.linalg.inv(sg_data)
		#-------------------------------------------------------------------------------

		#============= Transformations ====================================

		if transformation is "mas":
			Transformation = Iden

		elif transformation is "pc":
			Transformation = cartesianToSpherical_plus_mu

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
			pm.Uniform("location_0",lower=hyper_alpha[0][0],
									upper=hyper_alpha[0][1],
									shape=shape)
			pm.Uniform("location_1",lower=hyper_alpha[1][0],
									upper=hyper_alpha[1][1],
									shape=shape)
			pm.Uniform("location_2",lower=hyper_alpha[2][0],
									upper=hyper_alpha[2][1],
									shape=shape)
			pm.Uniform("location_3",lower=hyper_alpha[3][0],
									upper=hyper_alpha[3][1],
									shape=shape)
			pm.Uniform("location_4",lower=hyper_alpha[4][0],
									upper=hyper_alpha[4][1],
									shape=shape)

			#--------- Join variables --------------
			mu = pm.math.stack([self.location_0,
								self.location_1,
								self.location_2,
								self.location_3,
								self.location_4],axis=1)

		else:
			mu = parameters["location"]

		#------------------------ Scale ---------------------------------------
		if parameters["scale"] is None:
			pm.HalfCauchy("sd_0",beta=hyper_beta[0],shape=shape)
			pm.HalfCauchy("sd_1",beta=hyper_beta[1],shape=shape)
			pm.HalfCauchy("sd_2",beta=hyper_beta[2],shape=shape)
			pm.HalfCauchy("sd_3",beta=hyper_beta[3],shape=shape)
			pm.HalfCauchy("sd_4",beta=hyper_beta[4],shape=shape)

			# pm.LKJCorr('chol_corr', eta=hyper_gamma, n=5)
			# pm.Deterministic('C', tt.fill_diagonal(self.chol_corr[np.zeros((5, 5),
			#                    dtype=np.int64)], 1.))
			C = np.eye(5)

			sigma_diag  = pm.math.stack([self.sd_0,
										self.sd_1,
										self.sd_2,
										self.sd_3,
										self.sd_4],axis=1)

			cov = theano.shared(np.zeros((shape,5,5)))

			for i in range(shape):
				sigma       = tt.nlinalg.diag(sigma_diag[i])
				covi        = tt.nlinalg.matrix_dot(sigma, C, sigma)
				cov         = tt.set_subtensor(cov[i],covi)
			
			cov_print   = tt.printing.Print("cov")(cov)

		else:
			cov = parameters["scale"]
		#========================================================================

		#------------------ True values ---------------------------------------------
		if prior is "Gaussian":
			pm.MvNormal("source",mu=mu,cov=cov[0],shape=(N,5))

		elif prior is "GMM":
			pm.Dirichlet("weights",a=hyper_delta,shape=shape)

			comps = [ pm.MvNormal.dist(mu=mu[i],cov=cov[i]) for i in range(shape)] 

			pm.Mixture("source",w=self.weights,comp_dists=comps,shape=(N,5))
		
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


############################ 6D Model ###########################################################
class Model6D(pm.Model):
	'''
	Model to infer the position,distance and proper motions of a cluster
	'''
	def __init__(self,mu_data,sg_data,
		prior="Gaussian",
		parameters={"location":None,"scale":None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		transformation=None,
		name='flavour_6d', model=None):
		# 2) call super's init first, passing model and name
		# to it name will be prefix for all variables here if
		# no name specified for model there will be no prefix
		super().__init__(name, model)
		# now you are in the context of instance,
		# `modelcontext` will return self you can define
		# variables in several ways note, that all variables
		# will get model's name prefix

		#------------------- Data ------------------------------------------------------
		N = int(len(mu_data)/6)


		if N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")

		T = np.linalg.inv(sg_data)
		#-------------------------------------------------------------------------------

		#============= Transformations ====================================

		if transformation is "mas":
			Transformation = Iden

		elif transformation is "pc":
			Transformation = phaseSpaceToAstrometry

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
			pm.Uniform("location_0",lower=hyper_alpha[0][0],
									upper=hyper_alpha[0][1],
									shape=shape)
			pm.Uniform("location_1",lower=hyper_alpha[1][0],
									upper=hyper_alpha[1][1],
									shape=shape)
			pm.Uniform("location_2",lower=hyper_alpha[2][0],
									upper=hyper_alpha[2][1],
									shape=shape)
			pm.Uniform("location_3",lower=hyper_alpha[3][0],
									upper=hyper_alpha[3][1],
									shape=shape)
			pm.Uniform("location_4",lower=hyper_alpha[4][0],
									upper=hyper_alpha[4][1],
									shape=shape)
			pm.Uniform("location_5",lower=hyper_alpha[5][0],
									upper=hyper_alpha[5][1],
									shape=shape)

			#--------- Join variables --------------
			mu = pm.math.stack([self.location_0,
								 self.location_1,
								 self.location_2,
								 self.location_3,
								 self.location_4,
								 self.location_5],axis=1)

		else:
			mu = parameters["location"]

		#------------------------ Scale ---------------------------------------
		if parameters["scale"] is None:
			pm.HalfCauchy("sd_0",beta=hyper_beta[0],shape=shape)
			pm.HalfCauchy("sd_1",beta=hyper_beta[1],shape=shape)
			pm.HalfCauchy("sd_2",beta=hyper_beta[2],shape=shape)
			pm.HalfCauchy("sd_3",beta=hyper_beta[3],shape=shape)
			pm.HalfCauchy("sd_4",beta=hyper_beta[4],shape=shape)
			pm.HalfCauchy("sd_5",beta=hyper_beta[5],shape=shape)

			# pm.LKJCorr('chol_corr', eta=hyper_gamma, n=6)
			# pm.Deterministic('C', tt.fill_diagonal(self.chol_corr[np.zeros((6, 6),
			#                    dtype=np.int64)], 1.))
			C = np.eye(6)

			sigma_diag  = pm.math.stack([self.sd_0,
									self.sd_1,
									self.sd_2,
									self.sd_3,
									self.sd_4,
									self.sd_5],axis=1)

			cov = theano.shared(np.zeros((shape,6,6)))

			for i in range(shape):
				sigma       = tt.nlinalg.diag(sigma_diag[i])
				# sigma_print = tt.printing.Print("sigma")(sigma)
				covi        = tt.nlinalg.matrix_dot(sigma, C, sigma)
				cov         = tt.set_subtensor(cov[i],covi)
			
			# cov_print   = tt.printing.Print("cov")(cov[0])
			# mu_print    = tt.printing.Print("mu")(mu)
		   

		else:
			cov = parameters["scale"]
		#========================================================================

		#------------------ True values ---------------------------------------------
		if prior is "Gaussian":
			pm.MvNormal("source",mu=mu,cov=cov[0],shape=(N,6))

		elif prior is "GMM":
			pm.Dirichlet("weights",a=hyper_delta,shape=shape)

			comps = [ pm.MvNormal.dist(mu=mu[i],cov=cov[i]) for i in range(shape)] 

			pm.Mixture("source",w=self.weights,comp_dists=comps,shape=(N,6))
		
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