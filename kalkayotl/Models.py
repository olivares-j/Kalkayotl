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

PRINT = printing.Print('Theano shape: ', attrs=['shape'])

################################## Model 1D ####################################
class Model1D(Model):
	'''
	Model to infer the distance of a series of stars
	'''
	def __init__(self,n_sources,mu_data,tau_data,
		prior="Gaussian",
		parameters={"location":None,"scale": None},
		hyper_alpha=[100,10],
		hyper_beta=[10],
		hyper_gamma=None,
		hyper_delta=None,
		hyper_nu=None,
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

		elif prior == "StudentT":
			pm.Gamma("nu",alpha=hyper_nu["alpha"],beta=hyper_nu["beta"])
			if parametrization == "central":
				pm.StudentT("source",nu=self.nu,mu=self.loc,sd=self.scl,shape=n_sources)
			else:
				pm.StudentT("offset",nu=self.nu,mu=0.0,sd=1.0,shape=n_sources)
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
		hyper_nu=None,
		field_sd=None,
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

		#------------ Weights -----------------------------
		if "GMM" in prior:
			if parameters["weights"] is None:
				weights = pm.Dirichlet("weights",a=hyper_delta)
			else:
				weights = parameters["weights"]
		#---------------------------------------------------
		#==============================================================================

		#===================== True values ============================================		
		if prior == "Gaussian":
			if parametrization == "central":
				pm.MvNormal("source",mu=loc,chol=chol,shape=(n_sources,3))
			else:
				pm.Normal("offset",mu=0,sigma=1,shape=(n_sources,3))
				pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(self.offset,chol))

		elif prior == "StudentT":
			pm.Gamma("nu",alpha=hyper_nu["alpha"],beta=hyper_nu["beta"])
			if parametrization == "central":
				pm.MvStudentT("source",nu=self.nu,mu=loc,chol=chol,shape=(n_sources,3))
			else:
				pm.StudentT("offset",nu=self.nu,mu=0,sigma=1,shape=(n_sources,3))
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
			comps = [ pm.MvNormal.dist(mu=loc[i],chol=chol[i]) for i in range(n_components)]

			#---- Sample from the mixture ----------------------------------
			pm.Mixture("source",w=weights,comp_dists=comps,shape=(n_sources,3))

		elif prior == "FGMM":
			chol_field = np.diag(np.repeat(field_sd["position"],3))

			comps = [pm.MvNormal.dist(mu=loc,chol=chol),pm.MvNormal.dist(mu=loc,chol=chol_field)]

			#---- Sample from the mixture ----------------------------------
			pm.Mixture("source",w=weights,comp_dists=comps,shape=(n_sources,3))
		
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
		hyper_kappa=None,
		hyper_omega=None,
		hyper_nu=None,
		field_sd=None,
		transformation=None,
		reference_system="ICRS",
		parametrization="non-central",
		velocity_model="joint",
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

		######################### JOIN MODEL #########################################
		if velocity_model == "joint":
			print("Working with the joint velocity model.")
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

			#------------ Weights -----------------------------
			if "GMM" in prior:
				if parameters["weights"] is None:
					weights = pm.Dirichlet("weights",a=hyper_delta)
				else:
					weights = parameters["weights"]
			#---------------------------------------------------
			#==============================================================================

			#===================== True values ============================================		
			if prior == "Gaussian":
				if parametrization == "central":
					source = pm.MvNormal("source",mu=loc,chol=chol,shape=(n_sources,6))
				else:
					offset = pm.Normal("offset",mu=0,sigma=1,shape=(n_sources,6))
					source = pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(offset,chol))

			elif prior == "StudentT":
				nu = pm.Gamma("nu",alpha=hyper_nu["alpha"],beta=hyper_nu["beta"])
				if parametrization == "central":
					source = pm.MvStudentT("source",nu=nu,mu=loc,chol=chol,shape=(n_sources,6))
				else:
					offset = pm.StudentT("offset",nu=nu,mu=0,sigma=1,shape=(n_sources,6))
					source = pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(self.offset,chol))

			elif prior in ["GMM","CGMM"]:
				comps = [ pm.MvNormal.dist(mu=loc[i],chol=chol[i]) for i in range(n_components)]

				#---- Sample from the mixture ----------------------------------
				source = pm.Mixture("source",w=weights,comp_dists=comps,shape=(n_sources,6))

			elif prior == "FGMM":
				chol_field = np.diag(np.concatenate([
					np.repeat(field_sd["position"],3),
					np.repeat(field_sd["velocity"],3)]
					))

				comps = [pm.MvNormal.dist(mu=loc,chol=chol),pm.MvNormal.dist(mu=loc,chol=chol_field)]

				#---- Sample from the mixture ----------------------------------
				source = pm.Mixture("source",w=weights,comp_dists=comps,shape=(n_sources,6))
			
			else:
				sys.exit("The specified prior is not supported")
			#=================================================================================
		#############################################################################################

		##################### INDEPENDENT MODEL #####################################################
		elif velocity_model == "independent":
			print("Working with the independent velocity model")

			#================ Hyper-parameters =====================================
			#----------------- Mixture prior families ----------------------------
			if prior in ["GMM","CGMM"]:
				#------------- Shapes -------------------------
				n_components = len(hyper_delta)

				loc_pos  = theano.shared(np.zeros((n_components,3)))
				loc_vel  = theano.shared(np.zeros((n_components,3)))
				chol_pos = theano.shared(np.zeros((n_components,3,3)))
				chol_vel = theano.shared(np.zeros((n_components,3,3)))
				#----------------------------------------------

				#----------- Locations ------------------------------------------
				if parameters["location"] is None:
					if prior in ["CGMM"]:
						#----------------- Concentric prior --------------------
						location_pos = [ pm.Normal("loc_{0}".format(j),
									mu=hyper_alpha[j][0],
									sigma=hyper_alpha[j][1]) for j in range(3) ]

						location_vel = [ pm.Normal("loc_{0}".format(j),
									mu=hyper_alpha[j][0],
									sigma=hyper_alpha[j][1]) for j in range(3,6) ]

						loci_pos = pm.math.stack(location_pos,axis=1)
						loci_vel = pm.math.stack(location_vel,axis=1)

						for i in range(n_components):
							loc_pos  = tt.set_subtensor(loc_pos[i],loci_pos)
							loc_vel  = tt.set_subtensor(loc_vel[i],loci_vel)
						#---------------------------------------------------------

					else:
						#----------- Non-concentric prior ----------------------------
						location_pos = [ pm.Normal("loc_{0}".format(j),
									mu=hyper_alpha[j][0],
									sigma=hyper_alpha[j][1],
									shape=n_components) for j in range(3) ]

						location_vel = [ pm.Normal("loc_{0}".format(j),
									mu=hyper_alpha[j][0],
									sigma=hyper_alpha[j][1],
									shape=n_components) for j in range(3,6) ]
						
						loc_pos = pm.math.stack(location_pos,axis=1)
						loc_vel = pm.math.stack(location_vel,axis=1)
						#---------------------------------------------------------
					#-------------------------------------------------------------------
				else:
					for i in range(n_components):
						loc_pos  = tt.set_subtensor(loc_pos[i],
									np.array(parameters["location"][i][:3]))
						loc_vel  = tt.set_subtensor(loc_vel[i],
									np.array(parameters["location"][i][3:]))

				#---------- Covariance matrices -----------------------------------
				if parameters["scale"] is None:
					for i in range(n_components):
						choli_pos, corri_pos, stdsi_pos = pm.LKJCholeskyCov(
											"scl_pos_{0}".format(i), 
											n=3, eta=hyper_eta, 
											sd_dist=pm.Gamma.dist(
											alpha=2.0,beta=1.0/hyper_beta),
											compute_corr=True)
					
						chol_pos = tt.set_subtensor(chol_pos[i],choli_pos)

						choli_vel, corri_vel, stdsi_vel = pm.LKJCholeskyCov(
											"scl_vel_{0}".format(i), 
											n=3, eta=hyper_eta, 
											sd_dist=pm.Gamma.dist(
												alpha=2.0,
												beta=1.0/hyper_beta),
											compute_corr=True)
					
						chol_vel = tt.set_subtensor(chol_vel[i],choli_vel)

				else:
					for i in range(n_components):
						choli_pos = np.linalg.cholesky(parameters["scale"][i][:3])
						choli_vel = np.linalg.cholesky(parameters["scale"][i][3:])
						chol_pos = tt.set_subtensor(chol_pos[i],choli_pos)
						chol_vel = tt.set_subtensor(chol_vel[i],choli_vel)
				#--------------------------------------------------------------------
			#---------------------------------------------------------------------------------

			#-------------- Non-mixture prior families ----------------------------------
			else:
				#--------- Location ----------------------------------
				if parameters["location"] is None:
					location_pos = [ pm.Normal("loc_{0}".format(i),
								mu=hyper_alpha[i][0],
								sigma=hyper_alpha[i][1]) for i in range(3) ]

					location_vel = [ pm.Normal("loc_{0}".format(i),
								mu=hyper_alpha[i][0],
								sigma=hyper_alpha[i][1]) for i in range(3,6) ]

					#--------- Join variables --------------
					loc_pos = pm.math.stack(location_pos,axis=1)
					loc_vel = pm.math.stack(location_vel,axis=1)

				else:
					loc_pos = parameters["location"][:3]
					loc_vel = parameters["location"][3:]
				#------------------------------------------------------

				#---------- Covariance matrix ------------------------------------
				if parameters["scale"] is None:
					chol_pos, corr_pos, stds_pos = pm.LKJCholeskyCov(
										"scl_pos", 
										n=3, 
										eta=hyper_eta, 
										sd_dist=pm.Gamma.dist(
											alpha=2.0,
											beta=1.0/hyper_beta),
										compute_corr=True)

					chol_vel, corr_vel, stds_vel = pm.LKJCholeskyCov(
										"scl_vel", 
										n=3, 
										eta=hyper_eta, 
										sd_dist=pm.Gamma.dist(
											alpha=2.0,
											beta=1.0/hyper_beta),
										compute_corr=True)

				else:
					chol_pos = np.linalg.cholesky(parameters["scale"][:3])
					chol_vel = np.linalg.cholesky(parameters["scale"][3:])
				#--------------------------------------------------------------
			#----------------------------------------------------------------------------
			#==============================================================================

			#===================== True values ============================================		
			if prior == "Gaussian":
				if parametrization == "central":
					source_pos = pm.MvNormal("source_pos",mu=loc_pos,chol=chol_vel,shape=(n_sources,3))
					source_vel = pm.MvNormal("source_vel",mu=loc_vel,chol=chol_vel,shape=(n_sources,3))
				else:
					tau_pos = pm.Normal("tau_pos",mu=0,sigma=1,shape=(n_sources,3))
					tau_vel = pm.Normal("tau_vel",mu=0,sigma=1,shape=(n_sources,3))

					source_pos = pm.Deterministic("source_pos",
									loc_pos + tt.nlinalg.matrix_dot(tau_pos,chol_pos))
					source_vel = pm.Deterministic("source_vel",
									loc_vel + tt.nlinalg.matrix_dot(tau_vel,chol_vel))

			elif prior == "StudentT":
				nu = pm.Gamma("nu",alpha=hyper_nu["alpha"],beta=hyper_nu["beta"],shape=2)
				if parametrization == "central":
					source_pos = pm.MvStudentT("source_pos",nu=nu[0],mu=loc_pos,chol=chol_vel,shape=(n_sources,3))
					source_vel = pm.MvStudentT("source_vel",nu=nu[1],mu=loc_vel,chol=chol_vel,shape=(n_sources,3))
				else:
					tau_pos = pm.StudentT("tau_pos",nu=nu[0],mu=0,sigma=1,shape=(n_sources,3))
					tau_vel = pm.StudentT("tau_vel",nu=nu[1],mu=0,sigma=1,shape=(n_sources,3))

					source_pos = pm.Deterministic("source_pos",
									loc_pos + tt.nlinalg.matrix_dot(tau_pos,chol_pos))
					source_vel = pm.Deterministic("source_vel",
									loc_vel + tt.nlinalg.matrix_dot(tau_vel,chol_vel))

			elif prior in ["GMM","CGMM"]:
				comps_pos = [pm.MvNormal.dist(mu=loc_pos[i],chol=chol_pos[i]) for i in range(n_components)]
				comps_vel = [pm.MvNormal.dist(mu=loc_vel[i],chol=chol_vel[i]) for i in range(n_components)]

				#---- Sample from the mixture ----------------------------------
				source_pos = pm.Mixture("source_pos",w=weights,comp_dists=comps_pos,shape=(n_sources,3))
				source_vel = pm.Mixture("source_vel",w=weights,comp_dists=comps_vel,shape=(n_sources,3))

			elif prior == "FGMM":
				chol_field_pos = np.diag(np.repeat(field_sd["position"],3))
				chol_field_vel = np.diag(np.repeat(field_sd["velocity"],3))

				comps_pos = [pm.MvNormal.dist(mu=loc_pos,chol=chol_pos),
							pm.MvNormal.dist(mu=loc_pos,chol=chol_field_pos)]

				comps_vel = [pm.MvNormal.dist(mu=loc_vel,chol=chol_vel),
							pm.MvNormal.dist(mu=loc_vel,chol=chol_field_vel)]

				#---- Sample from the mixture ----------------------------------
				source_pos = pm.Mixture("source_pos",w=weights,comp_dists=comps_pos,shape=(n_sources,3))
				source_vel = pm.Mixture("source_vel",w=weights,comp_dists=comps_vel,shape=(n_sources,3))
			
			else:
				sys.exit("The specified prior is not supported")

			source = pm.Deterministic("source",tt.concatenate([source_pos,source_vel],axis=1))
			#=================================================================================

		#############################################################################################

		##################### LINEAR MODEL ###########################################
		elif velocity_model in ["linear","constant"]: 

			#================ Hyper-parameters =====================================
			#----------------- Mixture prior families ----------------------------
			if prior in ["GMM","CGMM"]:
				#------------- Shapes -------------------------
				n_components = len(hyper_delta)

				loc_pos  = theano.shared(np.zeros((n_components,3)))
				loc_vel  = theano.shared(np.zeros((n_components,3)))
				chol_pos = theano.shared(np.zeros((n_components,3,3)))
				chol_vel = theano.shared(np.zeros((n_components,3,3)))
				#----------------------------------------------

				#----------- Locations ------------------------------------------
				if parameters["location"] is None:
					if prior in ["CGMM"]:
						#----------------- Concentric prior --------------------
						location_pos = [ pm.Normal("loc_{0}".format(j),
									mu=hyper_alpha[j][0],
									sigma=hyper_alpha[j][1]) for j in range(3) ]

						location_vel = [ pm.Normal("loc_{0}".format(j),
									mu=hyper_alpha[j][0],
									sigma=hyper_alpha[j][1]) for j in range(3,6) ]

						loci_pos = pm.math.stack(location_pos,axis=1)
						loci_vel = pm.math.stack(location_vel,axis=1)

						for i in range(n_components):
							loc_pos  = tt.set_subtensor(loc_pos[i],loci_pos)
							loc_vel  = tt.set_subtensor(loc_vel[i],loci_vel)
						#---------------------------------------------------------

					else:
						#----------- Non-concentric prior ----------------------------
						location_pos = [ pm.Normal("loc_{0}".format(j),
									mu=hyper_alpha[j][0],
									sigma=hyper_alpha[j][1],
									shape=n_components) for j in range(3) ]

						location_vel = [ pm.Normal("loc_{0}".format(j),
									mu=hyper_alpha[j][0],
									sigma=hyper_alpha[j][1],
									shape=n_components) for j in range(3,6) ]
						
						loc_pos = pm.math.stack(location_pos,axis=1)
						loc_vel = pm.math.stack(location_vel,axis=1)
						#---------------------------------------------------------
					#-------------------------------------------------------------------
				else:
					for i in range(n_components):
						loc_pos  = tt.set_subtensor(loc_pos[i],
									np.array(parameters["location"][i][:3]))
						loc_vel  = tt.set_subtensor(loc_vel[i],
									np.array(parameters["location"][i][3:]))

				#---------- Covariance matrices -----------------------------------
				if parameters["scale"] is None:
					for i in range(n_components):
						choli_pos, corri_pos, stdsi_pos = pm.LKJCholeskyCov(
											"scl_pos_{0}".format(i), 
											n=3, eta=hyper_eta, 
											sd_dist=pm.Gamma.dist(
											alpha=2.0,beta=1.0/hyper_beta),
											compute_corr=True)
					
						chol_pos = tt.set_subtensor(chol_pos[i],choli_pos)

						choli_vel, corri_vel, stdsi_vel = pm.LKJCholeskyCov(
											"scl_vel_{0}".format(i), 
											n=3, eta=hyper_eta, 
											sd_dist=pm.Gamma.dist(
												alpha=2.0,
												beta=1.0/hyper_beta),
											compute_corr=True)
					
						chol_vel = tt.set_subtensor(chol_vel[i],choli_vel)

				else:
					for i in range(n_components):
						choli_pos = np.linalg.cholesky(parameters["scale"][i][:3])
						choli_vel = np.linalg.cholesky(parameters["scale"][i][3:])
						chol_pos = tt.set_subtensor(chol_pos[i],choli_pos)
						chol_vel = tt.set_subtensor(chol_vel[i],choli_vel)
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
					location_pos = [ pm.Normal("loc_{0}".format(i),
								mu=hyper_alpha[i][0],
								sigma=hyper_alpha[i][1]) for i in range(3) ]

					location_vel = [ pm.Normal("loc_{0}".format(i),
								mu=hyper_alpha[i][0],
								sigma=hyper_alpha[i][1]) for i in range(3,6) ]

					#--------- Join variables --------------
					loc_pos = pm.math.stack(location_pos,axis=1)
					loc_vel = pm.math.stack(location_vel,axis=1)

				else:
					loc_pos = parameters["location"][:3]
					loc_vel = parameters["location"][3:]
				#------------------------------------------------------

				#---------- Covariance matrix ------------------------------------
				if parameters["scale"] is None:
					chol_pos, corr_pos, stds_pos = pm.LKJCholeskyCov(
										"scl_pos", 
										n=3, 
										eta=hyper_eta, 
										sd_dist=pm.Gamma.dist(
											alpha=2.0,
											beta=1.0/hyper_beta),
										compute_corr=True)

					chol_vel, corr_vel, stds_vel = pm.LKJCholeskyCov(
										"scl_vel", 
										n=3, 
										eta=hyper_eta, 
										sd_dist=pm.Gamma.dist(
											alpha=2.0,
											beta=1.0/hyper_beta),
										compute_corr=True)

				else:
					chol_pos = np.linalg.cholesky(parameters["scale"][:3])
					chol_vel = np.linalg.cholesky(parameters["scale"][3:])
				#--------------------------------------------------------------
			#----------------------------------------------------------------------------
			#==============================================================================

			#=================== Velocity field ==============================
			if velocity_model == "constant":
				print("Working with the constant velocity model")
				kappa = pm.Normal("kappa",mu=0.0,sigma=hyper_kappa,shape=3)
				lnv = tt.nlinalg.diag(kappa)

			else:
				print("Working with the linear velocity model")
				kappa = pm.Normal("kappa",mu=0.0,sigma=hyper_kappa,shape=3)
				omega = pm.Normal("omega",mu=0.0,sigma=hyper_omega,shape=(2,3))

				lnv = tt.nlinalg.diag(kappa)
				lnv = tt.set_subtensor(lnv[np.triu_indices(3,1)],omega[0])
				lnv = tt.set_subtensor(lnv[np.tril_indices(3,-1)],omega[1])
			#=================================================================

			#===================== True values ============================================		
			if prior == "Gaussian":
				if parametrization == "central":
					offset_pos = pm.MvNormal("offset_pos",mu=0.0,chol=chol_pos,shape=(n_sources,3))
					offset_vel = pm.MvNormal("offset_vel",mu=0.0,chol=chol_vel,shape=(n_sources,3))
					offset_lnv = pm.Deterministic("offset_lnv",tt.nlinalg.matrix_dot(offset_pos,lnv))

					source_pos = pm.Deterministic("source_pos",loc_pos + offset_pos)
					source_vel = pm.Deterministic("source_vel",loc_vel + offset_lnv + offset_vel)

				else:
					tau_pos = pm.Normal("tau_pos",mu=0,sigma=1,shape=(n_sources,3))
					tau_vel = pm.Normal("tau_vel",mu=0,sigma=1,shape=(n_sources,3))

					offset_pos = pm.Deterministic("offset_pos",
									tt.nlinalg.matrix_dot(tau_pos,chol_pos))
					offset_vel = pm.Deterministic("offset_vel",
									tt.nlinalg.matrix_dot(tau_vel,chol_vel))
					offset_lnv = pm.Deterministic("offset_lnv",
									tt.nlinalg.matrix_dot(offset_pos,lnv))

					source_pos = pm.Deterministic("source_pos",loc_pos + offset_pos)
					source_vel = pm.Deterministic("source_vel",loc_vel + offset_lnv + offset_vel)
			
			else:
				sys.exit("The specified prior is not supported")

			source     = pm.Deterministic("source",tt.concatenate([source_pos,source_vel],axis=1))
			#=================================================================================
		###############################################################################################
		else:
			sys.exit("Velocity model not recognized!")


		#----------------------- Transformation---------------------------------------
		transformed = Transformation(source)
		#-----------------------------------------------------------------------------

		#------------ Flatten --------------------------------------------------------
		true = pm.math.flatten(transformed)
		#----------------------------------------------------------------------------

		#----------------------- Likelihood ----------------------------------------
		# Select only those observed e.g., true[idx_obs]
		pm.MvNormal('obs', mu=true[idx_data], tau=tau_data,observed=mu_data)
		#------------------------------------------------------------------------------