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

# from kalkayotl.Priors import EDSD,EFF,King

################################## Model 1D ####################################
class Model1D(Model):
	'''
	Model to infer the distance of a series of stars
	'''
	def __init__(self,
		n_sources,mu_data,tau_data,
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
				weights = pm.Dirichlet(parameters["weights"],
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

			#---------- Standard deviations ----------------------------------
			if parameters["scale"] is None and prior in ["GMM","CGMM"]:
				std = pm.Gamma("std",
							alpha=2.0,
							beta=1./hyper_beta,
							shape=(n_components,dimension),
							dims=("component","coordinate"))

			elif parameters["scale"] is None and prior == "FGMM":
				stds = pytensor.shared(np.zeros((n_components,dimension)))

				stds_i = pm.Gamma("stds_cls",
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
				std = tt.set_subtensor(std,np.array(parameters["scale"]))

				std = pm.Deterministic("std", std,dims=("component","coordinate"))
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
				loc = pm.Deterministic("loc",parameters["location"],
						dims="coordinate")
			#------------------------------------------------------

			#---------- Covariance matrix ------------------------------------
			if parameters["scale"] is None:
				std = pm.Gamma("std",
							alpha=2.0,
							beta=1./hyper_beta,
							shape=dimension,
							dims="coordinate")
			else:
				std = pm.Deterministic("std",parameters["scale"],
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

		#----------------------- Likelihood ----------------------------------------
		pm.MvNormal('obs', mu=pm.math.flatten(true), tau=tau_data,observed=mu_data)
		#------------------------------------------------------------------------------
####################################################################################################

############################ ND Model ###########################################################
class Model3D6D(Model):
	'''
	Model to infer the n_sources-dimensional parameter vector of a cluster
	'''
	def __init__(self,
		n_sources,
		mu_data,
		tau_data,
		idx_observed,
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
				weights = pm.Dirichlet(parameters["weights"],
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
												beta=hyper_beta),
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
												beta=hyper_beta),
											compute_corr=True,
											store_in_trace=False)
					chol = tt.set_subtensor(chol[i],chol_i)
					corr = tt.set_subtensor(corr[i],corr_i)
					stds = tt.set_subtensor(stds[i],stds_i)

				#------------ Field -----------------------------------
				chol_i = np.diag(parameters["field_scale"])
				cov = np.dot(chol_i, chol_i.T)
				stds_i = np.sqrt(np.diag(cov))
				inv_stds = 1. / stds_i
				corr_i = inv_stds[None, :] * cov * inv_stds[:, None]

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
					inv_stds = 1. / stds_i
					corr_i = inv_stds[None, :] * cov * inv_stds[:, None]
					#---------------------------------------------------

					chol = tt.set_subtensor(chol[i],chol_i)
					corr = tt.set_subtensor(corr[i],corr_i)
					stds = tt.set_subtensor(stds[i],stds_i)

			corr = pm.Deterministic("corr", corr,dims=("component","coordinate",))
			stds = pm.Deterministic("stds", stds,dims=("component","coordinate"))
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
				loc = pm.Deterministic("loc",parameters["location"],
						dims="coordinate")
			#------------------------------------------------------

			#---------- Covariance matrix ------------------------------------
			if parameters["scale"] is None:
				chol,corr,stds = pm.LKJCholeskyCov("chol", 
								n=dimension, 
								eta=hyper_eta, 
								sd_dist=pm.Gamma.dist(
									alpha=2.0,
									beta=hyper_beta),
								compute_corr=True,
								store_in_trace=False)
				corr = pm.Deterministic("corr", corr)
				stds = pm.Deterministic("stds", stds,
							dims="coordinate")
			else:
				sys.exit("Not yet implemented.")
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
				pm.Normal("offset",mu=0,sigma=1,
					shape=(n_sources,dimension))
				pm.Deterministic("source",
					loc + tt.nlinalg.matrix_dot(self.offset,chol),
					dims=("source_id","coordinate"))

		elif prior == "StudentT":
			pm.Gamma("nu",alpha=hyper_nu["alpha"],beta=hyper_nu["beta"])
			if parametrization == "central":
				pm.MvStudentT("source",nu=self.nu,mu=loc,chol=chol,
					shape=(n_sources,dimension),
					dims=("source_id","coordinate"))
			else:
				pm.StudentT("offset",nu=self.nu,mu=0,sigma=1,
					shape=(n_sources,dimension))
				pm.Deterministic("source",
					loc + tt.nlinalg.matrix_dot(self.offset,chol),
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

		#----------------------- Likelihood ----------------------------------------
		pm.MvNormal('obs',	mu=pm.math.flatten(true)[idx_observed], 
							tau=tau_data,
							observed=mu_data)
		#---------------------------------------------------------------------------

# class Model6D_independent(Model):
# 	'''
# 	Model to infer the 6-dimensional parameter vector of a cluster
# 	'''
# 	def __init__(self,
# 		n_sources,mu_data,tau_data,idx_data,
# 		prior="Gaussian",
# 		parameters={"location":None,"scale":None},
# 		hyper_alpha=None,
# 		hyper_beta=None,
# 		hyper_gamma=None,
# 		hyper_delta=None,
# 		hyper_eta=None,
# 		hyper_kappa=None,
# 		hyper_omega=None,
# 		hyper_nu=None,
# 		transformation=None,
# 		parametrization="non-central",
# 		name="6D",
# 		identifiers=None,
# 		coordinates=["X","Y","Z","U","V","W"],
# 		observables=["ra","dec","parallax","pmra","pmdec","radial_velocity"]):
# 		super().__init__(name, model=None)
# 		self.add_coord("source_id",values=identifiers)
# 		self.add_coord("coordinate",values=coordinates)
# 		self.add_coord("observable",values=observables)

# 		#------------------- Data ------------------------------------------------------
# 		if n_sources == 0:
# 			sys.exit("Data has length zero! You must provide at least one data point.")
# 		#-------------------------------------------------------------------------------

# 		print("Using {0} parametrization".format(parametrization))


# 		#================ Hyper-parameters =====================================
# 		#----------------- Mixture prior families ----------------------------
# 		if prior in ["GMM","CGMM"]:
# 			#------------- Shapes -------------------------
# 			n_components = len(hyper_delta)

# 			loc_pos  = pytensor.shared(np.zeros((n_components,3)))
# 			loc_vel  = pytensor.shared(np.zeros((n_components,3)))
# 			chol_pos = pytensor.shared(np.zeros((n_components,3,3)))
# 			chol_vel = pytensor.shared(np.zeros((n_components,3,3)))
# 			#----------------------------------------------

# 			#----------- Locations ------------------------------------------
# 			if parameters["location"] is None:
# 				if prior in ["CGMM"]:
# 					#----------------- Concentric prior --------------------
# 					location_pos = pm.Normal("loc_pos",
# 										mu=hyper_alpha["loc"][:3],
# 										sigma=hyper_alpha["scl"][:3],
# 										shape=3)

# 					location_vel = pm.Normal("loc_vel",
# 										mu=hyper_alpha["loc"][3:],
# 										sigma=hyper_alpha["scl"][3:],
# 										shape=3)

# 					for i in range(n_components):
# 						loc_pos  = tt.set_subtensor(loc_pos[i],loci_pos)
# 						loc_vel  = tt.set_subtensor(loc_vel[i],loci_vel)
# 					#---------------------------------------------------------

# 				else:
# 					#----------- Non-concentric prior ----------------------------
# 					location_pos = pm.Normal("loc_pos",
# 										mu=hyper_alpha["loc"][:3],
# 										sigma=hyper_alpha["scl"][:3],
# 										shape=(n_components,3))

# 					location_vel = pm.Normal("loc_vel",
# 										mu=hyper_alpha["loc"][3:],
# 										sigma=hyper_alpha["scl"][3:],
# 										shape=(n_components,3))
# 					#---------------------------------------------------------
# 				#-------------------------------------------------------------------
# 			else:
# 				for i in range(n_components):
# 					loc_pos  = tt.set_subtensor(loc_pos[i],
# 								np.array(parameters["location"][i][:3]))
# 					loc_vel  = tt.set_subtensor(loc_vel[i],
# 								np.array(parameters["location"][i][3:]))

# 			#---------- Covariance matrices -----------------------------------
# 			if parameters["scale"] is None:
# 				for i in range(n_components):
# 					choli_pos = pm.LKJCholeskyCov("chol_pos_{0}".format(i), 
# 										n=3, eta=hyper_eta, 
# 										sd_dist=pm.Gamma.dist(
# 											alpha=2.0,
# 											beta=hyper_beta),
# 										compute_corr=False,
# 										store_in_trace=False)
				
# 					chol_pos = tt.set_subtensor(chol_pos[i],choli_pos)

# 					choli_vel = pm.LKJCholeskyCov("chol_vel_{0}".format(i), 
# 										n=3, eta=hyper_eta, 
# 										sd_dist=pm.Gamma.dist(
# 											alpha=2.0,
# 											beta=hyper_beta),
# 										compute_corr=False,
# 										store_in_trace=False)
				
# 					chol_vel = tt.set_subtensor(chol_vel[i],choli_vel)

# 			else:
# 				for i in range(n_components):
# 					choli_pos = np.linalg.cholesky(parameters["scale"][i][:3])
# 					choli_vel = np.linalg.cholesky(parameters["scale"][i][3:])
# 					chol_pos = tt.set_subtensor(chol_pos[i],choli_pos)
# 					chol_vel = tt.set_subtensor(chol_vel[i],choli_vel)
# 			#--------------------------------------------------------------------
# 		#---------------------------------------------------------------------------------

# 		#-------------- Non-mixture prior families ----------------------------------
# 		else:
# 			#--------- Location ----------------------------------
# 			if parameters["location"] is None:
# 				location_pos = pm.Normal("loc_pos",
# 										mu=hyper_alpha["loc"][:3],
# 										sigma=hyper_alpha["scl"][:3],
# 										shape=3)

# 				location_vel = pm.Normal("loc_vel",
# 										mu=hyper_alpha["loc"][3:],
# 										sigma=hyper_alpha["scl"][3:],
# 										shape=3)

# 			else:
# 				loc_pos = parameters["location"][:3]
# 				loc_vel = parameters["location"][3:]
# 			#------------------------------------------------------

# 			#---------- Covariance matrix ------------------------------------
# 			if parameters["scale"] is None:
# 				chol_pos = pm.LKJCholeskyCov("chol_pos", 
# 									n=3, 
# 									eta=hyper_eta, 
# 									sd_dist=pm.Gamma.dist(
# 										alpha=2.0,
# 										beta=hyper_beta),
# 									compute_corr=False,
# 									store_in_trace=False)

# 				chol_vel = pm.LKJCholeskyCov("chol_vel", 
# 									n=3, 
# 									eta=hyper_eta, 
# 									sd_dist=pm.Gamma.dist(
# 										alpha=2.0,
# 										beta=hyper_beta),
# 									compute_corr=False,
# 									store_in_trace=False)

# 			else:
# 				chol_pos = np.linalg.cholesky(parameters["scale"][:3])
# 				chol_vel = np.linalg.cholesky(parameters["scale"][3:])
# 			#--------------------------------------------------------------
# 		#----------------------------------------------------------------------------
# 		#==============================================================================

# 		#===================== True values ============================================		
# 		if prior == "Gaussian":
# 			if parametrization == "central":
# 				source_pos = pm.MvNormal("source_pos",mu=loc_pos,chol=chol_vel,shape=(n_sources,3))
# 				source_vel = pm.MvNormal("source_vel",mu=loc_vel,chol=chol_vel,shape=(n_sources,3))
# 			else:
# 				tau_pos = pm.Normal("tau_pos",mu=0,sigma=1,shape=(n_sources,3))
# 				tau_vel = pm.Normal("tau_vel",mu=0,sigma=1,shape=(n_sources,3))

# 				source_pos = pm.Deterministic("source_pos",
# 								loc_pos + tt.nlinalg.matrix_dot(tau_pos,chol_pos))
# 				source_vel = pm.Deterministic("source_vel",
# 								loc_vel + tt.nlinalg.matrix_dot(tau_vel,chol_vel))

# 		elif prior == "StudentT":
# 			nu = pm.Gamma("nu",alpha=hyper_nu["alpha"],beta=hyper_nu["beta"],shape=2)
# 			if parametrization == "central":
# 				source_pos = pm.MvStudentT("source_pos",nu=nu[0],mu=loc_pos,chol=chol_vel,shape=(n_sources,3))
# 				source_vel = pm.MvStudentT("source_vel",nu=nu[1],mu=loc_vel,chol=chol_vel,shape=(n_sources,3))
# 			else:
# 				tau_pos = pm.StudentT("tau_pos",nu=nu[0],mu=0,sigma=1,shape=(n_sources,3))
# 				tau_vel = pm.StudentT("tau_vel",nu=nu[1],mu=0,sigma=1,shape=(n_sources,3))

# 				source_pos = pm.Deterministic("source_pos",
# 								loc_pos + tt.nlinalg.matrix_dot(tau_pos,chol_pos))
# 				source_vel = pm.Deterministic("source_vel",
# 								loc_vel + tt.nlinalg.matrix_dot(tau_vel,chol_vel))

# 		elif "GMM" in prior:
# 			comps_pos = [pm.MvNormal.dist(mu=loc_pos[i],chol=chol_pos[i]) for i in range(n_components)]
# 			comps_vel = [pm.MvNormal.dist(mu=loc_vel[i],chol=chol_vel[i]) for i in range(n_components)]

# 			#---- Sample from the mixture ----------------------------------
# 			source_pos = pm.Mixture("source_pos",w=weights,comp_dists=comps_pos,shape=(n_sources,3))
# 			source_vel = pm.Mixture("source_vel",w=weights,comp_dists=comps_vel,shape=(n_sources,3))
		
# 		else:
# 			sys.exit("The specified prior is not supported")

# 		source = pm.Deterministic("source",
# 					tt.concatenate([source_pos,source_vel],axis=1),
# 					dims=("source_id","coordinate"))
# 		#=================================================================================

# 		#----------------------- Transformation-----------------------
# 		true = pm.Deterministic("true",Transformation(source),
# 					dims=("source_id","observable"))
# 		#-------------------------------------------------------------

# 		#----------------------- Likelihood --------------------------------------
# 		pm.MvNormal('obs', mu=pm.math.flatten(true)[idx_data], 
# 					tau=tau_data,observed=mu_data)
# 		#-------------------------------------------------------------------------

class Model6D_linear(Model):
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
		hyper_kappa=None,
		hyper_omega=None,
		hyper_nu=None,
		transformation=None,
		parametrization="central",
		velocity_model="linear",
		identifiers=None,
		coordinates=["X","Y","Z","U","V","W"],
		observables=["ra","dec","parallax","pmra","pmdec","radial_velocity"]):
		super().__init__(name="6D", model=None)
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
				weights = pm.Dirichlet(parameters["weights"],
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
												beta=hyper_beta),
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
												beta=hyper_beta),
											compute_corr=True,
											store_in_trace=False)
					chol = tt.set_subtensor(chol[i],chol_i)
					corr = tt.set_subtensor(corr[i],corr_i)
					stds = tt.set_subtensor(stds[i],stds_i)

				#------------ Field -----------------------------------
				chol_i = np.diag(parameters["field_scale"])
				cov = np.dot(chol_i, chol_i.T)
				stds_i = np.sqrt(np.diag(cov))
				inv_stds = 1. / stds_i
				corr_i = inv_stds[None, :] * cov * inv_stds[:, None]

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
					inv_stds = 1. / stds_i
					corr_i = inv_stds[None, :] * cov * inv_stds[:, None]
					#---------------------------------------------------

					chol = tt.set_subtensor(chol[i],chol_i)
					corr = tt.set_subtensor(corr[i],corr_i)
					stds = tt.set_subtensor(stds[i],stds_i)

			corr = pm.Deterministic("corr", corr,dims=("component","coordinate",))
			stds = pm.Deterministic("stds", stds,dims=("component","coordinate"))
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
				loc = pm.Deterministic("loc",parameters["location"],
						dims="coordinate")
			#------------------------------------------------------

			#---------- Covariance matrix ------------------------------------
			if parameters["scale"] is None:
				chol_pos,corr_pos,stds_pos = pm.LKJCholeskyCov("chol_pos", 
								n=3, 
								eta=hyper_eta, 
								sd_dist=pm.Gamma.dist(
									alpha=2.0,
									beta=hyper_beta[:3]),
								compute_corr=True,
								store_in_trace=False)
				chol_vel,corr_vel,stds_vel = pm.LKJCholeskyCov("chol_vel", 
								n=3, 
								eta=hyper_eta, 
								sd_dist=pm.Gamma.dist(
									alpha=2.0,
									beta=hyper_beta[3:]),
								compute_corr=True,
								store_in_trace=False)

				corr_pos = pm.Deterministic("corr_pos", corr_pos)
				corr_vel = pm.Deterministic("corr_vel", corr_vel)

			else:
				sys.exit("Not yet implemented.")
			#--------------------------------------------------------------

			stds = pm.Deterministic("stds",
						tt.concatenate([stds_pos,stds_vel],axis=0),
						dims="coordinate")
		#----------------------------------------------------------------------------
		#==============================================================================

		#=================== Velocity field ==============================
		lnv = pytensor.shared(np.zeros((3,3)))
		kappa = pm.Normal("kappa",mu=0.0,sigma=hyper_kappa,shape=3,dims="coordinate")
		if velocity_model == "linear":
			print("Working with the linear velocity model")
			omega = pm.Normal("omega",mu=0.0,sigma=hyper_omega,shape=(2,3))
			lnv = tt.set_subtensor(lnv[np.triu_indices(3,1)],omega[0])
			lnv = tt.set_subtensor(lnv[np.tril_indices(3,-1)],omega[1])
		else:
			print("Working with the constant velocity model")

		lnv = tt.set_subtensor(lnv[np.diag_indices(3)],kappa)
		#=================================================================

		#===================== True values ============================================		
		if prior == "Gaussian":
			if parametrization == "central":
				offset_pos = pm.MvNormal("offset_pos",mu=0.0,chol=chol_pos,shape=(n_sources,3))
				offset_vel = pm.MvNormal("offset_vel",mu=0.0,chol=chol_vel,shape=(n_sources,3))
			else:
				tau = pm.Normal("tau",mu=0,sigma=1,shape=(n_sources,6))

				offset_pos = tt.nlinalg.matrix_dot(tau[:,:3],chol_pos)
				offset_vel = tt.nlinalg.matrix_dot(tau[:,3:],chol_vel)

			source_pos = loc[:3] + offset_pos
			source_vel = loc[3:] + offset_vel + tt.nlinalg.matrix_dot(offset_pos,lnv)

		elif prior == "StudentT":
			nu = pm.Gamma("nu",alpha=hyper_nu["alpha"],beta=hyper_nu["beta"],shape=2)
			if parametrization == "central":
				offset_pos = pm.MvStudentT("offset_pos",nu=nu[0],mu=0.0,chol=chol_pos,shape=(n_sources,3))
				offset_vel = pm.MvStudentT("offset_vel",nu=nu[1],mu=0.0,chol=chol_vel,shape=(n_sources,3))

			else:
				tau_pos = pm.StudentT("tau_pos",nu=nu[0],mu=0,sigma=1,shape=(n_sources,3))
				tau_vel = pm.StudentT("tau_vel",nu=nu[1],mu=0,sigma=1,shape=(n_sources,3))

				offset_pos = tt.nlinalg.matrix_dot(tau[:,:3],chol_pos)
				offset_vel = tt.nlinalg.matrix_dot(tau[:,3:],chol_vel)

			source_pos = loc[:3] + offset_pos
			source_vel = loc[3:] + offset_vel + tt.nlinalg.matrix_dot(offset_pos,lnv)
		
		else:
			sys.exit("The specified prior is not yet supported")

		source = pm.Deterministic("source",
						tt.concatenate([source_pos,source_vel],axis=1),
						dims=("source_id","coordinate"))
		#=================================================================================

		#----------------------- Transformation-----------------------
		true = pm.Deterministic("true",transformation(source),
					dims=("source_id","observable"))
		#-------------------------------------------------------------

		#----------------------- Likelihood --------------------------------------
		pm.MvNormal('obs', mu=pm.math.flatten(true)[idx_data], 
					tau=tau_data,observed=mu_data)
		#-------------------------------------------------------------------------