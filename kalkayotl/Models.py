'''
Copyright 2020 Javier Olivares Romero

This file is part of Kalkayotl.

	Kalkayotl is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Kalkayotl is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Kalkayotl.  If not, see <http://www.gnu.org/licenses/>.
'''
import sys
import numpy as np
import pymc3 as pm
from pymc3 import Model
import theano
from theano import tensor as tt, printing

from kalkayotl.Transformations import Iden,pc2mas # 1D
from kalkayotl.Transformations import icrs_xyz_to_radecplx,galactic_xyz_to_radecplx #3D
from kalkayotl.Transformations import icrs_xyzuvw_to_astrometry_and_rv
from kalkayotl.Transformations import galactic_xyzuvw_to_astrometry_and_rv #6D
from kalkayotl.Priors import EDSD,MvEFF #,EFF,King,MvEFF,MvKing

################################## Model 1D ####################################
class Model1D(Model):
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
		name="1D", model=None):
		super().__init__(name, model)

		#------------------- Data ------------------------------------------------------
		if n_sources == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")

		#-------------------------------------------------------------------------------

		#============= Transformations ====================================

		if transformation == "mas":
			Transformation = Iden

		elif transformation == "pc":
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
			pm.Gamma("scl",alpha=2.0,beta=2.0/hyper_beta,shape=shape)
		else:
			self.scl = parameters["scale"]
		#========================================================================

		#================= True values ========================================================
		#--------- Cluster oriented prior-----------------------------------------------
		if prior == "Uniform":
			if parametrization == "central":
				pm.Uniform("source",lower=self.loc-self.scl,upper=self.loc+self.scl,shape=n_sources)
			else:
				pm.Uniform("offset",lower=-1.,upper=1.,shape=n_sources)
				pm.Deterministic("source",self.loc + self.scl*self.offset)

		elif prior == "Gaussian":
			if parametrization == "central":
				pm.Normal("source",mu=self.loc,sd=self.scl,shape=n_sources)
			else:
				pm.Normal("offset",mu=0.0,sd=1.0,shape=n_sources)
				pm.Deterministic("source",self.loc + self.scl*self.offset)

		elif prior == "GMM":
			# break symmetry and avoids inf in advi
			pm.Potential('order_means', tt.switch(self.loc[1]-self.loc[0] < 0, -1e20, 0))

			if parameters["weights"] is None:
				pm.Dirichlet("weights",a=hyper_delta,shape=shape)	
			else:
				self.weights = parameters["weights"]

			if parametrization == "central":
				pm.NormalMixture("source",w=self.weights,
					mu=self.loc,
					sigma=self.scl,
					comp_shape=1,
					shape=n_sources)
			else:
				pm.Normal("offset",mu=0.0,sd=1.0,shape=n_sources)
				# latent cluster of each observation
				component = pm.Categorical("component",p=self.weights,shape=n_sources)
				pm.Deterministic("source",self.loc[component] + self.scl[component]*self.offset) 

		elif prior == "EFF":
			if parameters["gamma"] is None:
				pm.Gamma("x",alpha=2.0,beta=2.0/hyper_gamma)
				pm.Deterministic("gamma",1.0+self.x)
			else:
				self.gamma = parameters["gamma"]

			if parametrization == "central":
				EFF("source",location=self.loc,scale=self.scl,gamma=self.gamma,shape=n_sources)
			else:
				EFF("offset",location=0.0,scale=1.0,gamma=self.gamma,shape=n_sources)
				pm.Deterministic("source",self.loc + self.scl*self.offset)

		elif prior == "King":
			if parameters["rt"] is None:
				pm.Gamma("x",alpha=2.0,beta=2.0/hyper_gamma)
				pm.Deterministic("rt",1.0+self.x)
			else:
				self.rt = parameters["rt"]

			if parametrization == "central":
				King("source",location=self.loc,scale=self.scl,rt=self.rt,shape=n_sources)
			else:
				King("offset",location=0.0,scale=1.0,rt=self.rt,shape=n_sources)
				pm.Deterministic("source",self.loc + self.scl*self.offset)
			
		#---------- Galactic oriented prior ---------------------------------------------
		elif prior == "EDSD":
			EDSD("source",scale=self.scl,shape=n_sources)
		
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
class Model3D(Model):
	'''
	Model to infer the n_sources-dimensional parameter vector of a cluster
	'''
	def __init__(self,n_sources,mu_data,tau_data,
		prior="Gaussian",
		parameters={"location":None,"scale":None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		hyper_eta=None,
		transformation=None,
		reference_system="ICRS",
		parametrization="non-central",
		name="3D", model=None):
		super().__init__(name, model)

		#------------------- Data ------------------------------------------------------
		if n_sources == 0:
			sys.exit("Data has length zero! You must provide at least one data point.")
		#-------------------------------------------------------------------------------

		print("Using {0} parametrization".format(parametrization))

		#============= Transformations ====================================
		assert transformation == "pc","3D model only works with 'pc' transformation"

		if reference_system == "ICRS":
				Transformation = icrs_xyz_to_radecplx
		elif reference_system == "Galactic":
				Transformation = galactic_xyz_to_radecplx
		else:
			sys.exit("Reference system not accepted")
		#==================================================================

		#================ Hyper-parameters =====================================
		#----------------- Mixture prior families ----------------------------
		if prior in ["GMM","CGMM"]:
			#------------- Shapes -------------------------
			n_components = len(hyper_delta)

			loc  = theano.shared(np.zeros((n_components,3)))
			chol = theano.shared(np.zeros((n_components,3,3)))
			#----------------------------------------------

			#----------- Locations ------------------------------------------
			if parameters["location"] is None:
				if prior in ["CGMM"]:
					#----------------- Concentric prior --------------------
					location = [ pm.Normal("loc_{0}".format(j),
								mu=hyper_alpha[j][0],
								sigma=hyper_alpha[j][1]) for j in range(3) ]

					loci = pm.math.stack(location,axis=1)

					for i in range(n_components):
						loc  = tt.set_subtensor(loc[i],loci)
					#---------------------------------------------------------

				else:
					#----------- Non-concentric prior ----------------------------
					location = [ pm.Normal("loc_{0}".format(j),
								mu=hyper_alpha[j][0],
								sigma=hyper_alpha[j][1],
								shape=n_components) for j in range(3) ]
					
					loc = pm.math.stack(location,axis=1)
					#---------------------------------------------------------
				#-------------------------------------------------------------------
			else:
				for i in range(n_components):
					loc  = tt.set_subtensor(loc[i],np.array(parameters["location"][i]))

			#---------- Covariance matrices -----------------------------------
			if parameters["scale"] is None:
				for i in range(n_components):
					choli, corri, stdsi = pm.LKJCholeskyCov("scl_{0}".format(i), 
										n=3, eta=hyper_eta, 
										sd_dist=pm.Gamma.dist(
										alpha=2.0,beta=1.0/hyper_beta),
										compute_corr=True)
				
					chol = tt.set_subtensor(chol[i],choli)

			else:
				sys.exit("Not yet implemented.")
			#--------------------------------------------------------------------
		#---------------------------------------------------------------------------------

		#-------------- Non-mixture prior families ----------------------------------
		else:
			#--------- Location ----------------------------------
			if parameters["location"] is None:
				location = [ pm.Normal("loc_{0}".format(i),
							mu=hyper_alpha[i][0],
							sigma=hyper_alpha[i][1]) for i in range(3) ]

				#--------- Join variables --------------
				loc = pm.math.stack(location,axis=1)

			else:
				loc = parameters["location"]
			#------------------------------------------------------

			#---------- Covariance matrix ------------------------------------
			if parameters["scale"] is None:
				chol, corr, stds = pm.LKJCholeskyCov("scl", n=3, eta=hyper_eta, 
						sd_dist=pm.Gamma.dist(alpha=2.0,beta=1.0/hyper_beta),
						compute_corr=True)
			else:
				sys.exit("Not yet implemented.")
			#--------------------------------------------------------------
		#----------------------------------------------------------------------------
		#==============================================================================

		#===================== True values ============================================		
		if prior == "Gaussian":
			if parametrization == "central":
				pm.MvNormal("source",mu=loc,chol=chol,shape=(n_sources,3))
			else:
				pm.Normal("offset",mu=0,sigma=1,shape=(n_sources,3))
				pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(self.offset,chol))

		elif prior == "King":
			if parameters["rt"] is None:
				pm.Gamma("x",alpha=2.0,beta=1.0/hyper_gamma)
				pm.Deterministic("rt",1.001+self.x)
			else:
				self.rt = parameters["rt"]

			if parametrization == "central":
				MvKing("source",location=loc,chol=chol,rt=self.rt,shape=(n_sources,3))
			else:
				MvKing("offset",location=np.zeros(3),chol=np.eye(3),rt=self.rt,shape=(n_sources,3))
				pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(self.offset,chol))

		elif prior == "EFF":
			if parameters["gamma"] is None:
				pm.Gamma("x",alpha=2.0,beta=1.0/hyper_gamma)
				pm.Deterministic("gamma",3.001+self.x )
			else:
				self.gamma = parameters["gamma"]

			if parametrization == "central":
				MvEFF("source",location=loc,chol=chol,gamma=self.gamma,shape=(n_sources,3))
			else:
				MvEFF("offset",location=np.zeros(3),chol=np.eye(3),gamma=self.gamma,shape=(n_sources,3))
				pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(self.offset,chol))

		elif prior in ["GMM","CGMM"]:
			pm.Dirichlet("weights",a=hyper_delta)

			comps = [ pm.MvNormal.dist(mu=loc[i],chol=chol[i]) for i in range(n_components)]

			#---- Sample from the mixture ----------------------------------
			pm.Mixture("source",w=self.weights,comp_dists=comps,shape=(n_sources,3))
		
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

############################ 6D Model ###########################################################
class Model6D(Model):
	'''
	Model to infer the 6-dimensional parameter vector of a cluster
	'''
	def __init__(self,n_sources,mu_data,tau_data,idx_data,
		prior="Gaussian",
		parameters={"location":None,"scale":None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		hyper_eta=None,
		transformation=None,
		reference_system="ICRS",
		parametrization="non-central",
		name="6D", model=None):
		super().__init__(name, model)

		#------------------- Data ------------------------------------------------------
		if n_sources == 0:
			sys.exit("Data has length zero! You must provide at least one data point.")
		#-------------------------------------------------------------------------------

		print("Using {0} parametrization".format(parametrization))

		#============= Transformations ==============================================
		assert transformation == "pc","6D model only works with 'pc' transformation"

		if reference_system == "ICRS":
			Transformation = icrs_xyzuvw_to_astrometry_and_rv
		elif reference_system == "Galactic":
			Transformation = galactic_xyzuvw_to_astrometry_and_rv
		else:
			sys.exit("Reference system not accepted")
		#===========================================================================

		#================ Hyper-parameters =====================================
		#----------------- Mixture prior families ----------------------------
		if prior in ["GMM","CGMM"]:
			#------------- Shapes -------------------------
			n_components = len(hyper_delta)

			loc  = theano.shared(np.zeros((n_components,6)))
			chol = theano.shared(np.zeros((n_components,6,6)))
			#----------------------------------------------

			#----------- Locations ------------------------------------------
			if parameters["location"] is None:
				if prior in ["CGMM"]:
					#----------------- Concentric prior --------------------
					location = [ pm.Normal("loc_{0}".format(j),
								mu=hyper_alpha[j][0],
								sigma=hyper_alpha[j][1]) for j in range(6) ]

					loci = pm.math.stack(location,axis=1)

					for i in range(n_components):
						loc  = tt.set_subtensor(loc[i],loci)
					#---------------------------------------------------------

				else:
					#----------- Non-concentric prior ----------------------------
					location = [ pm.Normal("loc_{0}".format(j),
								mu=hyper_alpha[j][0],
								sigma=hyper_alpha[j][1],
								shape=n_components) for j in range(6) ]
					
					loc = pm.math.stack(location,axis=1)
					#---------------------------------------------------------
				#-------------------------------------------------------------------
			else:
				for i in range(n_components):
					loc  = tt.set_subtensor(loc[i],np.array(parameters["location"][i]))

			#---------- Covariance matrices -----------------------------------
			if parameters["scale"] is None:
				for i in range(n_components):
					choli, corri, stdsi = pm.LKJCholeskyCov("scl_{0}".format(i), 
										n=6, eta=hyper_eta, 
										sd_dist=pm.Gamma.dist(
										alpha=2.0,beta=1.0/hyper_beta),
										compute_corr=True)
				
					chol = tt.set_subtensor(chol[i],choli)

			else:
				for i in range(n_components):
					choli = np.linalg.cholesky(parameters["scale"][i])
					chol = tt.set_subtensor(chol[i],choli)
			#--------------------------------------------------------------------

			#------------ Weights -----------------------------
			if parameters["weights"] is None:
				weights = pm.Dirichlet("weights",a=hyper_delta)
			else:
				weights = parameters["weights"]
			#---------------------------------------------------
		#---------------------------------------------------------------------------------

		#-------------- Non-mixture prior families ----------------------------------
		else:
			#--------- Location ----------------------------------
			if parameters["location"] is None:
				location = [ pm.Normal("loc_{0}".format(i),
							mu=hyper_alpha[i][0],
							sigma=hyper_alpha[i][1]) for i in range(6) ]

				#--------- Join variables --------------
				loc = pm.math.stack(location,axis=1)

			else:
				loc = parameters["location"]
			#------------------------------------------------------

			#---------- Covariance matrix ------------------------------------
			if parameters["scale"] is None:
				chol, corr, stds = pm.LKJCholeskyCov("scl", n=6, eta=hyper_eta, 
						sd_dist=pm.Gamma.dist(alpha=2.0,beta=1.0/hyper_beta),
						compute_corr=True)
			else:
				chol = np.linalg.cholesky(parameters["scale"])
			#--------------------------------------------------------------
		#----------------------------------------------------------------------------
		#==============================================================================

		#===================== True values ============================================		
		if prior == "Gaussian":
			if parametrization == "central":
				pm.MvNormal("source",mu=loc,chol=chol,shape=(n_sources,6))
			else:
				pm.Normal("offset",mu=0,sigma=1,shape=(n_sources,6))
				pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(self.offset,chol))

		elif prior in ["GMM","CGMM"]:
			comps = [ pm.MvNormal.dist(mu=loc[i],chol=chol[i]) for i in range(n_components)]

			#---- Sample from the mixture ----------------------------------
			pm.Mixture("source",w=weights,comp_dists=comps,shape=(n_sources,6))
		
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
		# Select only those observed e.g., true[idx_obs]
		pm.MvNormal('obs', mu=true[idx_data], tau=tau_data,observed=mu_data)
		#------------------------------------------------------------------------------