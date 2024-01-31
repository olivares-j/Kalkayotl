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
import pymc as pm
import string
from pymc import Model
import pytensor
from pytensor import tensor as tt, function,printing,pp

from kalkayotl.GGD import GeneralizedGamma


################################## Model 1D ####################################
class Model1D(Model):
	'''
	Model to infer the distance of a series of stars
	'''
	def __init__(self,n_sources,mu_data,tau_data,
		indep_measures=False,
		dimension=1,
		prior="Gaussian",
		parameters={"location":None,"scale": None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		hyper_nu=None,
		transformation=None,
		parametrization="central",
		identifiers=None,
		coordinates=["distance"],
		observables=["parallax"]
		):
		super().__init__(name="1D", model=None)
		self.add_coord("source_id",values=identifiers)
		self.add_coord("coordinate",values=coordinates)
		self.add_coord("observable",values=observables)

		print("Using {0} parametrization".format(parametrization))
		assert dimension == 1, "This class is only for 1D models!"

		#================ Hyper-parameters =====================================
		#----------------- Mixture prior families ----------------------------
		if "GMM" in prior:
			#------------- Names ---------------------------------------------------
			n_components = len(hyper_delta)
			if prior == "FGMM":
				names_components = list(string.ascii_uppercase)[:(n_components-1)]
				names_components.append("Field")
			else:
				names_components = list(string.ascii_uppercase)[:n_components]
			self.add_coord("component",values=names_components)
			#-----------------------------------------------------------------------

			#------------- Weights ---------------------------
			if parameters["weights"] is None:
				weights = pm.Dirichlet("weights",
							a=hyper_delta,dims="component")
			else:
				weights = pm.Deterministic("weights",pytensor.shared(parameters["weights"]),
							dims="component")
			#------------------------------------------------

			#----------- Locations ------------------------------------------
			if parameters["location"] is None:
				if prior == "GMM":
					loc = pm.Normal("loc",
								mu=hyper_alpha["loc"],
								sigma=hyper_alpha["scl"],
								shape=(n_components,dimension),
								dims=("component","coordinate"))
				elif prior == "GGD":
					alpha = pm.Uniform("alpha",lower=0.0,upper=100.,initval=hyper_alpha)
					beta = pm.Uniform("beta",lower=-1.0,upper=99.,initval=hyper_gamma)
				else:
					#-------------- Repeat same location --------------
					loc_i = pm.Normal("centre",
								mu=hyper_alpha["loc"],
								sigma=hyper_alpha["scl"],
								shape=dimension)

					loc  = pytensor.shared(np.zeros((n_components,dimension)))
					for i in range(n_components):
						loc  = tt.set_subtensor(loc[i],loc_i)

					loc = pm.Deterministic("loc",loc,
						dims=("component","coordinate"))
					#--------------------------------------------------
					
			else:
				#----------------- Fixed location -----------------
				loc  = pytensor.shared(np.zeros((n_components,dimension)))
				for i in range(n_components):
					loc  = tt.set_subtensor(loc[i],
							np.array(parameters["location"][i]))
				loc = pm.Deterministic("loc",loc,
					dims=("component","coordinate"))
				#--------------------------------------------------
			#----------------------------------------------------------------------------

			#---------- Standard deviations ----------------------------------
			if parameters["scale"] is None and prior in ["GMM","CGMM"]:
				std = pm.Gamma("std",
							alpha=2.0,
							beta=1./hyper_beta,
							shape=(n_components,dimension),
							dims=("component","coordinate"))

			elif parameters["scale"] is None and prior == "FGMM":
				stds = pytensor.shared(np.zeros((n_components,dimension)))

				stds_i = pm.Gamma("sds_cls",
							alpha=2.0,
							beta=1./hyper_beta,
							shape=(n_components-1,dimension))

				stds = tt.set_subtensor(stds[:(n_components-1)],stds_i)

				#------------ Field -----------------------------------
				stds = tt.set_subtensor(stds[-1],
						np.array(parameters["field_scale"]))
				#---------------------------------------------------

				std = pm.Deterministic("std", stds,dims=("component","coordinate"))

			else:
				std = pytensor.shared(np.zeros((n_components,dimension)))
				for i in range(n_components):
					std = tt.set_subtensor(std[i],np.array(parameters["scale"][i]))

				std = pm.Deterministic("std", std, dims=("component","coordinate"))
			#--------------------------------------------------------------------
		#---------------------------------------------------------------------------------

		#-------------- Non-mixture prior families ----------------------------------
		else:
			#--------- Location ----------------------------------
			if parameters["location"] is None:
				loc = pm.Normal("loc",
						mu=hyper_alpha["loc"],
						sigma=hyper_alpha["scl"],
						shape=dimension,
						dims="coordinate")
			else:
				loc = pm.Deterministic("loc",
						pytensor.shared(parameters["location"]),
						dims="coordinate")
			#------------------------------------------------------

			#---------- Covariance matrix ------------------------------------
			if parameters["scale"] is None:
				std = pm.Gamma("std",
							alpha=2.0,
							beta=1./hyper_beta,
							shape=dimension,
							dims="coordinate",
							initval=[200.])
			else:
				std = pm.Deterministic("std",
						pytensor.shared(parameters["scale"]),
						dims="coordinate")
			#--------------------------------------------------------------
		#----------------------------------------------------------------------------
		#==============================================================================

		#================= True values ========================================================
		#--------- Cluster oriented prior-----------------------------------------------
		if prior == "Uniform":
			if parametrization == "central":
				source = pm.Uniform("source",lower=loc-std,upper=loc+std,
									shape=(n_sources,dimension),
									dims=("source_id","coordinate"))
			else:
				offset = pm.Uniform("offset",lower=-1.,upper=1.,shape=(n_sources,dimension))
				source = pm.Deterministic("source",loc + std*offset,
									dims=("source_id","coordinate"))

		elif prior == "Gaussian":
			if parametrization == "central":
				source = pm.Normal("source",mu=loc,sigma=std,shape=(n_sources,dimension),
									dims=("source_id","coordinate"))
			else:
				offset = pm.Normal("offset",mu=0.0,sigma=1.0,shape=(n_sources,dimension))
				source = pm.Deterministic("source",loc + std*offset,
									dims=("source_id","coordinate"))

		elif prior == "StudentT":
			nu = pm.Gamma("nu",alpha=hyper_nu["alpha"],beta=hyper_nu["beta"])

			if parametrization == "central":
				source = pm.StudentT("source",nu=nu,mu=loc,sigma=std,shape=(n_sources,dimension),
									dims=("source_id","coordinate"))
			else:
				offset = pm.StudentT("offset",nu=nu,mu=0.0,sigma=1.0,shape=(n_sources,dimension))
				source = pm.Deterministic("source",loc + std*offset,
									dims=("source_id","coordinate"))

		# elif prior == "EFF":
		# 	if parameters["gamma"] is None:
		# 		x = pm.Gamma("x",alpha=2.0,beta=1./hyper_gamma)
		# 		gamma = pm.Deterministic("gamma",1.0+x)
		# 	else:
		# 		gamma = pytensor.shared(np.array(parameters["gamma"]))

		# 	if parametrization == "central":
		# 		source = EFF("source",location=loc,scale=std,gamma=gamma,
		# 							shape=(n_sources,dimension),
		# 							dims=("source_id","coordinate"))
		# 	else:
		# 		offset = EFF("offset",location=0.0,scale=1.0,gamma=gamma,
		# 							shape=(n_sources,dimension))
		# 		source = pm.Deterministic("source",loc + std*offset,
		# 							dims=("source_id","coordinate"))

		# elif prior == "King":
		# 	if parameters["rt"] is None:
		# 		x  = pm.Gamma("x",alpha=2.0,beta=1./hyper_gamma)
		# 		rt = pm.Deterministic("rt",1.0+x)
		# 	else:
		# 		rt = pytensor.shared(np.array(parameters["rt"]))

		# 	if parametrization == "central":
		# 		source = King("source",location=loc,scale=scl,rt=rt,
		# 							shape=(n_sources,dimension),
		# 							dims=("source_id","coordinate"))
		# 	else:
		# 		offset = King("offset",location=0.0,scale=1.0,rt=rt,
		# 							shape=(n_sources,dimension))
		# 		source = pm.Deterministic("source",loc + std*offset,
		# 							dims=("source_id","coordinate"))
			
		# elif prior == "EDSD":
		# 	source = EDSD("source",scale=std,
		# 							shape=(n_sources,dimension),
		# 							dims=("source_id","coordinate"))
		# elif prior == "GGD": #NOTE: hyper_beta goes to the scale parameter, so alpha=hyper_alpha, beta=hyper_gamma
		# 	source = GGD("source",scale=std,alpha=alpha,beta=beta,
		# 							shape=(n_sources,dimension),
		# 							dims=("source_id","coordinate"))

		elif "GMM" in prior:				
			comps = [ pm.Normal.dist(mu=loc[i],sigma=std[i]) for i in range(n_components)]

			source = pm.Mixture("source",w=weights,comp_dists=comps,
									shape=(n_sources,dimension),
									dims=("source_id","coordinate"))

		else:
			sys.exit("ERROR: prior not recognized!")
		#-----------------------------------------------------------------------------
		#=======================================================================================

		#----------------- Transformations -------------------------
		true = pm.Deterministic("true",transformation(source),
									dims=("source_id","observable"))

		#----------------------- Likelihood --------------------------------------
		if indep_measures:
			pm.Normal('obs', mu=pm.math.flatten(true), 
						sigma=tau_data,observed=mu_data)
		else:
			pm.MvNormal('obs', mu=pm.math.flatten(true), 
						chol=tau_data,observed=mu_data)
		#-------------------------------------------------------------------------
####################################################################################################

############################ ND Model ###########################################################
class Model3D6D(Model):
	'''
	Model to infer the n_sources-dimensional parameter vector of a cluster
	'''
	def __init__(self,n_sources,mu_data,tau_data,idx_data,
		indep_measures=False,
		dimension=3,
		prior="Gaussian",
		parameters={"location":None,"scale":None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		hyper_eta=None,
		hyper_nu=None,
		transformation=None,
		parametrization="non-central",
		identifiers=None,
		coordinates=["X","Y","Z"],
		observables=["ra","dec","parallax"]):
		super().__init__(name="{0}D".format(dimension),model=None)
		self.add_coord("source_id",values=identifiers)
		self.add_coord("coordinate",values=coordinates)
		self.add_coord("observable",values=observables)

		#------------------- Data ------------------------------------------------------
		if n_sources == 0:
			sys.exit("Data has length zero! You must provide at least one data point.")
		#-------------------------------------------------------------------------------

		print("Using {0} parametrization".format(parametrization))

		#================ Hyper-parameters =====================================
		#----------------- Mixture prior families ----------------------------
		if "GMM" in prior:
			#------------- Names ---------------------------------------------------
			n_components = len(hyper_delta)
			if prior == "FGMM":
				names_components = list(string.ascii_uppercase)[:(n_components-1)]
				names_components.append("Field")
			else:
				names_components = list(string.ascii_uppercase)[:n_components]
			self.add_coord("component",values=names_components)
			#-----------------------------------------------------------------------

			#------------- Weights ---------------------------
			if parameters["weights"] is None:
				weights = pm.Dirichlet("weights",
							a=hyper_delta,dims="component")
			else:
				weights = pm.Deterministic("weights",
							pytensor.shared(parameters["weights"]),
							dims="component")
			#------------------------------------------------

			#----------- Locations ------------------------------------------
			if parameters["location"] is None:
				if prior == "GMM":
					loc = pm.Normal("loc",
								mu=hyper_alpha["loc"],
								sigma=hyper_alpha["scl"],
								shape=(n_components,dimension),
								dims=("component","coordinate"))
				else:
					#-------------- Repeat same location --------------
					loc_i = pm.Normal("centre",
								mu=hyper_alpha["loc"],
								sigma=hyper_alpha["scl"],
								shape=dimension)

					loc  = pytensor.shared(np.zeros((n_components,dimension)))
					for i in range(n_components):
						loc  = tt.set_subtensor(loc[i],loc_i)

					loc = pm.Deterministic("loc",loc,
						dims=("component","coordinate"))
					#--------------------------------------------------
					
			else:
				#----------------- Fixed location -----------------
				loc  = pytensor.shared(np.zeros((n_components,dimension)))
				for i in range(n_components):
					loc  = tt.set_subtensor(loc[i],
							np.array(parameters["location"][i]))
				loc = pm.Deterministic("loc",loc,
					dims=("component","coordinate"))
				#--------------------------------------------------
			#----------------------------------------------------------------------------

			#---------- Covariance matrices -----------------------------------
			stds = pytensor.shared(np.zeros((n_components,dimension)))
			corr = pytensor.shared(np.zeros((n_components,dimension,dimension)))
			chol = pytensor.shared(np.zeros((n_components,dimension,dimension)))

			if parameters["scale"] is None and prior in ["GMM","CGMM"]:
				for i,name in enumerate(names_components):
					chol_i,corr_i,stds_i = pm.LKJCholeskyCov(
											"chol_{0}".format(name), 
											n=dimension, 
											eta=hyper_eta, 
											sd_dist=pm.Gamma.dist(
												alpha=2.0,
												beta=1./hyper_beta),
											compute_corr=True,
											store_in_trace=False)
					chol = tt.set_subtensor(chol[i],chol_i)
					corr = tt.set_subtensor(corr[i],corr_i)
					stds = tt.set_subtensor(stds[i],stds_i)

			elif parameters["scale"] is None and prior == "FGMM":
				for i in range(n_components-1):
					chol_i,corr_i,stds_i = pm.LKJCholeskyCov(
											"chol_{0}".format(names_components[i]), 
											n=dimension, 
											eta=hyper_eta, 
											sd_dist=pm.Gamma.dist(
												alpha=2.0,
												beta=1./hyper_beta),
											compute_corr=True,
											store_in_trace=False)
					chol = tt.set_subtensor(chol[i],chol_i)
					corr = tt.set_subtensor(corr[i],corr_i)
					stds = tt.set_subtensor(stds[i],stds_i)

				#------------ Field -----------------------------------
				stds_i = np.array(parameters["field_scale"])
				chol_i = np.diag(stds_i)
				cov = np.dot(chol_i, chol_i.T)
				inv_stds = np.diag(1. / stds_i)
				corr_i = inv_stds @ cov @ inv_stds

				chol = tt.set_subtensor(chol[-1],chol_i)
				corr = tt.set_subtensor(corr[-1],corr_i)
				stds = tt.set_subtensor(stds[-1],stds_i)
				#---------------------------------------------------

			else:
				for i,name in enumerate(names_components):
					#--------- Extract ---------------------------------
					chol_i = np.linalg.cholesky(parameters["scale"][i])
					cov = np.dot(chol_i, chol_i.T)
					stds_i = np.sqrt(np.diag(cov))
					inv_stds = np.diag(1. / stds_i)
					corr_i = inv_stds @ cov @ inv_stds
					#---------------------------------------------------

					chol = tt.set_subtensor(chol[i],chol_i)
					corr = tt.set_subtensor(corr[i],corr_i)
					stds = tt.set_subtensor(stds[i],stds_i)

			corr = pm.Deterministic("corr", corr,dims=("component","coordinate",))
			stds = pm.Deterministic("std", stds,dims=("component","coordinate"))
			#--------------------------------------------------------------------
		#---------------------------------------------------------------------------------

		#-------------- Non-mixture prior families ----------------------------------
		else:
			#--------- Location ----------------------------------
			if parameters["location"] is None:
				loc = pm.Normal("loc",
						mu=hyper_alpha["loc"],
						sigma=hyper_alpha["scl"],
						shape=dimension,
						dims="coordinate")
			else:
				loc = pm.Deterministic("loc",pytensor.shared(parameters["location"]),
						dims="coordinate")
			#------------------------------------------------------

			#---------- Covariance matrix ------------------------------------
			if parameters["scale"] is None:
				chol,corr,stds = pm.LKJCholeskyCov("chol", 
								n=dimension, 
								eta=hyper_eta, 
								sd_dist=pm.Gamma.dist(
									alpha=2.0,
									beta=1./hyper_beta),
								compute_corr=True,
								store_in_trace=False)
				corr = pm.Deterministic("corr", corr)
				stds = pm.Deterministic("std", stds,
							dims="coordinate")
			else:
				#--------- Extract ---------------------------------
				chol_i = np.linalg.cholesky(parameters["scale"])
				cov = np.dot(chol_i, chol_i.T)
				stds_i = np.sqrt(np.diag(cov))
				inv_stds = np.diag(1. / stds_i)
				corr_i = inv_stds @ cov @ inv_stds
				#---------------------------------------------------
				
				chol = pytensor.shared(chol_i)
				corr = pm.Deterministic("corr", pytensor.shared(corr_i))
				stds = pm.Deterministic("std",  pytensor.shared(stds_i),
												dims="coordinate")
			#--------------------------------------------------------------
		#----------------------------------------------------------------------------
		#==============================================================================

		#===================== True values ============================================		
		if prior == "Gaussian":
			if parametrization == "central":
				pm.MvNormal("source",mu=loc,chol=chol,
					shape=(n_sources,dimension),
					dims=("source_id","coordinate"))
			else:
				offset = pm.Normal("offset",mu=0,sigma=1,
					shape=(n_sources,dimension))
				pm.Deterministic("source",loc + chol.dot(offset.T).T,
					dims=("source_id","coordinate"))

		elif prior == "StudentT":
			nu = pm.Gamma("nu",alpha=hyper_nu["alpha"],beta=hyper_nu["beta"])
			if parametrization == "central":
				pm.MvStudentT("source",nu=nu,mu=loc,chol=chol,
					shape=(n_sources,dimension),
					dims=("source_id","coordinate"))
			else:
				offset = pm.StudentT("offset",nu=nu,mu=0,sigma=1,
					shape=(n_sources,dimension))
				pm.Deterministic("source",loc + chol.dot(offset.T).T,
					dims=("source_id","coordinate"))

		# elif prior == "King":
		# 	if parameters["rt"] is None:
		# 		pm.Gamma("x",alpha=2.0,beta=1.0/hyper_gamma)
		# 		pm.Deterministic("rt",1.001+self.x)
		# 	else:
		# 		self.rt = parameters["rt"]

		# 	if parametrization == "central":
		# 		MvKing("source",location=loc,chol=chol,rt=self.rt,
		# 			shape=(n_sources,dimension),
		# 			dims=("source_id","coordinate"))
		# 	else:
		# 		MvKing("offset",location=np.zeros(dimension),chol=np.eye(dimension),rt=self.rt,
		# 			shape=(n_sources,dimension))
		# 		pm.Deterministic("source",
		# 			loc + tt.nlinalg.matrix_dot(self.offset,chol),
		# 			dims=("source_id","coordinate"))

		# elif prior == "EFF":
		# 	if parameters["gamma"] is None:
		# 		pm.Gamma("x",alpha=2.0,beta=1.0/hyper_gamma)
		# 		pm.Deterministic("gamma",dimension.001+self.x )
		# 	else:
		# 		self.gamma = parameters["gamma"]

		# 	if parametrization == "central":
		# 		MvEFF("source",location=loc,chol=chol,gamma=self.gamma,
		# 			shape=(n_sources,dimension),
		# 			dims=("source_id","coordinate"))
		# 	else:
		# 		MvEFF("offset",location=np.zeros(dimension),chol=np.eye(dimension),gamma=self.gamma,
		# 			shape=(n_sources,dimension))
		# 		pm.Deterministic("source",
		# 			loc + tt.nlinalg.matrix_dot(self.offset,chol),
		# 			dims=("source_id","coordinate"))

		elif "GMM" in prior:
			comps = [ pm.MvNormal.dist(mu=loc[i],chol=chol[i]) for i in range(n_components)]

			#---- Sample from the mixture ----------------------------------
			pm.Mixture("source",w=weights,comp_dists=comps,shape=(n_sources,dimension),
					dims=("source_id","coordinate"))
		
		else:
			sys.exit("The specified prior is not supported")
		#=================================================================================

		#----------------------- Transformation-----------------------
		true = pm.Deterministic("true",transformation(self.source),
					dims=("source_id","observable"))
		#-------------------------------------------------------------

		#----------------------- Likelihood --------------------------------------
		if indep_measures:
			pm.Normal('obs', mu=pm.math.flatten(true)[idx_data], 
						sigma=tau_data,observed=mu_data)
		else:
			pm.MvNormal('obs', mu=pm.math.flatten(true)[idx_data], 
						chol=tau_data,observed=mu_data)
		#-------------------------------------------------------------------------

class Model6D_linear(Model):
	'''
	Model to infer the 6-dimensional parameter vector of a cluster
	'''
	def __init__(self,n_sources,mu_data,tau_data,idx_data,
		indep_measures=False,
		prior="Gaussian",
		parameters={"location":None,"scale":None,"kappa":None,"omega":None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		hyper_eta=None,
		hyper_kappa=None,
		hyper_omega=None,
		hyper_nu=None,
		hyper_tau=None,
		hyper_age=None,
		transformation=None,
		parametrization="central",
		velocity_model="linear",
		identifiers=None,
		coordinates=["X","Y","Z","U","V","W"],
		observables=["ra","dec","parallax","pmra","pmdec","radial_velocity"]):
		super().__init__(name="6D", model=None)
		self.add_coord("source_id",values=identifiers)
		self.add_coord("coordinate",values=coordinates)
		self.add_coord("positions",values=["X","Y","Z"])
		self.add_coord("observable",values=observables)

		#------------------- Data ------------------------------------------------------
		if n_sources == 0:
			sys.exit("Data has length zero! You must provide at least one data point.")
		#-------------------------------------------------------------------------------

		print("Using {0} parametrization".format(parametrization))

		#================ Hyper-parameters =====================================
		#----------------- Mixture prior families ----------------------------
		if "GMM" in prior:
			sys.exit("Not yet implemented")
			#------------- Names ---------------------------------------------------
			n_components = len(hyper_delta)
			if prior == "FGMM":
				names_components = list(string.ascii_uppercase)[:(n_components-1)]
				names_components.append("Field")
			else:
				names_components = list(string.ascii_uppercase)[:n_components]
			self.add_coord("component",values=names_components)
			#-----------------------------------------------------------------------

			#------------- Weights ---------------------------
			if parameters["weights"] is None:
				weights = pm.Dirichlet("weights",
							a=hyper_delta,dims="component")
			else:
				weights = pm.Deterministic("weights",pytensor.shared(parameters["weights"]),
							dims="component")
			#------------------------------------------------

			#----------- Locations ------------------------------------------
			if parameters["location"] is None:
				if prior == "GMM":
					loc = pm.Normal("loc",
								mu=hyper_alpha["loc"],
								sigma=hyper_alpha["scl"],
								shape=(n_components,6),
								dims=("component","coordinate"))
				else:
					#-------------- Repeat same location --------------
					loc_i = pm.Normal("centre",
								mu=hyper_alpha["loc"],
								sigma=hyper_alpha["scl"],
								shape=6)

					loc  = pytensor.shared(np.zeros((n_components,6)))
					for i in range(n_components):
						loc  = tt.set_subtensor(loc[i],loc_i)

					loc = pm.Deterministic("loc",loc,
						dims=("component","coordinate"))
					#--------------------------------------------------
					
			else:
				#----------------- Fixed location -----------------
				loc  = pytensor.shared(np.zeros((n_components,6)))
				for i in range(n_components):
					loc  = tt.set_subtensor(loc[i],
							np.array(parameters["location"][i]))
				loc = pm.Deterministic("loc",loc,
					dims=("component","coordinate"))
				#--------------------------------------------------
			#----------------------------------------------------------------------------

			#---------- Covariance matrices -----------------------------------
			stds = pytensor.shared(np.zeros((n_components,6)))
			corr = pytensor.shared(np.zeros((n_components,6,6)))
			chol = pytensor.shared(np.zeros((n_components,6,6)))

			if parameters["scale"] is None and prior in ["GMM","CGMM"]:
				for i,name in enumerate(names_components):
					chol_i,corr_i,stds_i = pm.LKJCholeskyCov(
											"chol_{0}".format(name), 
											n=dimension, 
											eta=hyper_eta, 
											sd_dist=pm.Gamma.dist(
												alpha=2.0,
												beta=1./hyper_beta),
											compute_corr=True,
											store_in_trace=False)
					chol = tt.set_subtensor(chol[i],chol_i)
					corr = tt.set_subtensor(corr[i],corr_i)
					stds = tt.set_subtensor(stds[i],stds_i)

			elif parameters["scale"] is None and prior == "FGMM":
				for i in range(n_components-1):
					chol_i,corr_i,stds_i = pm.LKJCholeskyCov(
											"chol_{0}".format(names_components[i]), 
											n=dimension, 
											eta=hyper_eta, 
											sd_dist=pm.Gamma.dist(
												alpha=2.0,
												beta=1./hyper_beta),
											compute_corr=True,
											store_in_trace=False)
					chol = tt.set_subtensor(chol[i],chol_i)
					corr = tt.set_subtensor(corr[i],corr_i)
					stds = tt.set_subtensor(stds[i],stds_i)

				#------------ Field -----------------------------------
				stds_i = np.array(parameters["field_scale"])
				chol_i = np.diag(stds_i)
				cov = np.dot(chol_i, chol_i.T)
				inv_stds = np.diag(1. / stds_i)
				corr_i = inv_stds @ cov @ inv_stds

				chol = tt.set_subtensor(chol[-1],chol_i)
				corr = tt.set_subtensor(corr[-1],corr_i)
				stds = tt.set_subtensor(stds[-1],stds_i)
				#---------------------------------------------------

			else:
				for i,name in enumerate(names_components):
					#--------- Extract ---------------------------------
					chol_i = np.linalg.cholesky(parameters["scale"][i])
					cov = np.dot(chol_i, chol_i.T)
					stds_i = np.sqrt(np.diag(cov))
					inv_stds = np.diag(1. / stds_i)
					corr_i = inv_stds @ cov @ inv_stds
					#---------------------------------------------------

					chol = tt.set_subtensor(chol[i],chol_i)
					corr = tt.set_subtensor(corr[i],corr_i)
					stds = tt.set_subtensor(stds[i],stds_i)

			corr = pm.Deterministic("corr", corr,dims=("component","coordinate",))
			stds = pm.Deterministic("std", stds,dims=("component","coordinate"))
			#--------------------------------------------------------------------
		#---------------------------------------------------------------------------------

		#-------------- Non-mixture prior families ----------------------------------
		else:
			#--------- Location ----------------------------------
			if parameters["location"] is None:
				loc = pm.Normal("loc",
						mu=hyper_alpha["loc"],
						sigma=hyper_alpha["scl"],
						shape=6,
						dims="coordinate")
			else:
				loc = pm.Deterministic("loc",pytensor.shared(parameters["location"]),
						dims="coordinate")
			#------------------------------------------------------

			#---------- Covariance matrix ------------------------------------
			if parameters["scale"] is None:
				chol_pos,corr_pos,stds_pos = pm.LKJCholeskyCov("chol_pos", 
								n=3, 
								eta=hyper_eta, 
								sd_dist=pm.Gamma.dist(
									alpha=2.0,
									beta=1./hyper_beta[:3]),
								compute_corr=True,
								store_in_trace=False)
				chol_vel,corr_vel,stds_vel = pm.LKJCholeskyCov("chol_vel", 
								n=3, 
								eta=hyper_eta, 
								sd_dist=pm.Gamma.dist(
									alpha=2.0,
									beta=1./hyper_beta[3:]),
								compute_corr=True,
								store_in_trace=False)

				corr_pos = pm.Deterministic("corr_pos", corr_pos)
				corr_vel = pm.Deterministic("corr_vel", corr_vel)

			else:
				#------------- Extract scale of positions -------------------
				chol_pos_i = np.linalg.cholesky(parameters["scale"][:3,:3])
				cov = np.dot(chol_pos_i, chol_pos_i.T)
				stds_pos_i = np.sqrt(np.diag(cov))
				inv_stds = np.diag(1. / stds_pos_i)
				corr_pos_i = inv_stds @ cov @ inv_stds
				#----------------------------------------------------------

				#------------- Extract scale of velocities -------------------
				chol_vel_i = np.linalg.cholesky(parameters["scale"][3:,3:])
				cov = np.dot(chol_vel_i, chol_vel_i.T)
				stds_vel_i = np.sqrt(np.diag(cov))
				inv_stds = np.diag(1. / stds_vel_i)
				corr_vel_i = inv_stds @ cov @ inv_stds
				#----------------------------------------------------------
				
				chol_pos = pytensor.shared(chol_pos_i)
				chol_vel = pytensor.shared(chol_vel_i)

				stds_pos = pytensor.shared(stds_pos_i)
				stds_vel = pytensor.shared(stds_vel_i)

				corr_pos = pm.Deterministic("corr_pos", pytensor.shared(corr_pos_i))
				corr_vel = pm.Deterministic("corr_vel", pytensor.shared(corr_vel_i))
				
			#--------------------------------------------------------------

			stds = pm.Deterministic("std",
						tt.concatenate([stds_pos,stds_vel],axis=0),
						dims="coordinate")
		#----------------------------------------------------------------------------
		#==============================================================================

		#=================== Velocity field ==============================
		lnv = pytensor.shared(np.zeros((3,3)))

		#-------------------------- Kappa ----------------------------------------
		if parameters["kappa"] is None:
			if "age" in parameters.keys():
				if parameters["age"] is None:
					age = pm.TruncatedNormal("age",
											mu=hyper_age["loc"],
											sigma=hyper_age["scl"],
											lower=0.0,
											upper=hyper_age["upper"])
				else:
					age = pm.Deterministic("age",pytensor.shared(parameters["age"]))

				if hyper_tau["d"] is None:
					tau_d = pm.Gamma("tau_d",alpha=2,beta=hyper_tau["hyper_tau_d_beta"])
				else:
					tau_d = pm.Deterministic("tau_d",pytensor.shared(hyper_tau["d"]))

				if hyper_tau["p"] is None:
					tau_p = pm.Gamma("tau_p",alpha=2,beta=hyper_tau["hyper_tau_p_beta"])
				else:
					tau_p = pm.Deterministic("tau_p",pytensor.shared(hyper_tau["p"]))

				tau = GeneralizedGamma("tau",a=age,d=tau_d,p=tau_p,dims="positions")
				kappa = pm.Deterministic("kappa",1.0227121683768/tau,dims="positions")
			else:
				kappa = pm.Normal("kappa",
							mu=hyper_kappa["loc"],
							sigma=hyper_kappa["scl"],
							dims="positions")
		else:
			kappa = pm.Deterministic("kappa",
							pytensor.shared(parameters["kappa"]),
							dims="positions")
		#-------------------------------------------------------------------------
		
		if velocity_model == "linear":
			print("Working with the linear velocity model")

			#-------------------------- Omega ----------------------------------------
			if parameters["omega"] is None:
				omega = pm.Normal("omega",mu=0.0,sigma=hyper_omega,shape=(2,3))
			else:
				omega = pm.Deterministic("omega",pytensor.shared(parameters["omega"]))
			#-------------------------------------------------------------------------
			
			lnv = tt.set_subtensor(lnv[np.triu_indices(3,1)],omega[0])
			lnv = tt.set_subtensor(lnv[np.tril_indices(3,-1)],omega[1])
		else:
			print("Working with the constant velocity model")

		lnv = tt.set_subtensor(lnv[np.diag_indices(3)],kappa)
		#=================================================================

		#===================== True values =========================================================================	
		if prior == "Gaussian":
			if parametrization == "central":
				source_pos = pm.MvNormal("source_pos",mu=loc[:3],chol=chol_pos,shape=(n_sources,3))
				jitter_vel = pm.MvNormal("jitter_vel",mu=loc[3:],chol=chol_vel,shape=(n_sources,3))

				offset_pos = source_pos - loc[:3]
			else:
				epsilon = pm.Normal("epsilon",mu=0,sigma=1,shape=(n_sources,6))

				jitter_vel = loc[3:] + chol_vel.dot(epsilon[:,3:].T).T
				offset_pos = chol_pos.dot(epsilon[:,:3].T).T
				source_pos = loc[:3] + offset_pos
				
			source_vel = jitter_vel + lnv.dot(offset_pos.T).T

		elif prior == "StudentT":
			nu = pm.Gamma("nu",alpha=hyper_nu["alpha"],beta=hyper_nu["beta"],shape=2)
			if parametrization == "central":
				source_pos = pm.MvStudentT("source_pos",nu=nu[0],mu=loc[:3],chol=chol_pos,shape=(n_sources,3))
				jitter_vel = pm.MvStudentT("jitter_vel",nu=nu[1],mu=loc[3:],chol=chol_vel,shape=(n_sources,3))

				offset_pos = source_pos - loc[:3]
			else:
				epsilon = pm.StudentT("epsilon",nu=nu[0],mu=0,sigma=1,shape=(n_sources,6))

				jitter_vel = loc[3:] + chol_vel.dot(epsilon[:,3:].T).T
				offset_pos = chol_pos.dot(epsilon[:,:3].T).T
				source_pos = loc[:3] + offset_pos
				
			source_vel = jitter_vel + lnv.dot(offset_pos.T).T
		
		else:
			sys.exit("The specified prior is not yet supported")

		source = pm.Deterministic("source",
						tt.concatenate([source_pos,source_vel],axis=1),
						dims=("source_id","coordinate"))
		#=========================================================================================================

		#----------------------- Transformation-----------------------
		true = pm.Deterministic("true",transformation(source),
					dims=("source_id","observable"))
		#-------------------------------------------------------------

		#----------------------- Likelihood --------------------------------------
		if indep_measures:
			pm.Normal('obs', mu=pm.math.flatten(true)[idx_data], 
						sigma=tau_data,observed=mu_data)
		else:
			pm.MvNormal('obs', mu=pm.math.flatten(true)[idx_data], 
						chol=tau_data,observed=mu_data)
		#-------------------------------------------------------------------------
