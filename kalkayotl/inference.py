'''
Copyright 2019 Javier Olivares Romero

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
from __future__ import absolute_import, unicode_literals, print_function
import os
import sys
import random
import pymc as pm
import numpy as np
import pandas as pn
import xarray
import arviz as az
import h5py
import dill
import scipy.stats as st
from scipy.linalg import inv as inverse
from string import ascii_uppercase
from astropy.stats import circmean
from astropy import units as u
import pymc.sampling_jax
import pytensor.tensor as at
from typing import cast
import string

#---------------- Matplotlib -------------------------------------
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from matplotlib.colors import TwoSlopeNorm,Normalize
from matplotlib import lines as mlines
import matplotlib.ticker as ticker
#------------------------------------------------------------------

#------------ Local libraries ------------------------------------------
from kalkayotl.Models import Model1D,Model3D6D,Model6D_linear
from kalkayotl.Functions import AngularSeparation,CovarianceParallax,CovariancePM,get_principal,my_mode
# from kalkayotl.Evidence import Evidence1D
from kalkayotl.Transformations import astrometry_and_rv_to_phase_space
from kalkayotl.Transformations import Iden,pc2mas,mas2pc # 1D
from kalkayotl.Transformations import icrs_xyz_to_radecplx,galactic_xyz_to_radecplx #3D
from kalkayotl.Transformations import np_radecplx_to_icrs_xyz,np_radecplx_to_galactic_xyz #3D
from kalkayotl.Transformations import icrs_xyzuvw_to_astrometry_and_rv #6D
from kalkayotl.Transformations import galactic_xyzuvw_to_astrometry_and_rv #6D
from kalkayotl.Transformations import np_astrometry_and_rv_to_icrs_xyzuvw #6D
from kalkayotl.Transformations import np_astrometry_and_rv_to_galactic_xyzuvw #6D
#------------------------------------------------------------------------


class Inference:
	"""
	This class provides flexibility to infer the distance distribution given the parallax and its uncertainty
	"""
	def __init__(self,dimension,
				dir_out,
				zero_points,
				indep_measures=False,
				reference_system="Galactic",
				sampling_space="physical",
				velocity_model="joint",
				id_name='source_id',
				precision=2,
				**kwargs):
		"""
		Arguments:
		dimension (integer):  Dimension of the inference
		prior (string):       Prior family
		parameters (dict):    Prior parameters( location and scale)
		hyper_alpha (matrix)  Hyper-parameters of location
		hyper_beta (list)     Hyper-parameters of scale
		hyper_gamma (vector)  Hyper-parameters of weights (only for GMM prior)    
		"""
		np.set_printoptions(precision=precision,suppress=True)
		gaia_observables = ["ra","dec","parallax","pmra","pmdec","radial_velocity",
					"ra_error","dec_error","parallax_error","pmra_error","pmdec_error","radial_velocity_error",
					"ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr",
					"dec_parallax_corr","dec_pmra_corr","dec_pmdec_corr",
					"parallax_pmra_corr","parallax_pmdec_corr",
					"pmra_pmdec_corr"]

		coordinates = ["X","Y","Z","U","V","W"]

		assert dimension in [1,3,6], "Dimension must be 1, 3 or 6"
		assert isinstance(zero_points,dict), "zero_points must be a dictionary"
		assert reference_system in ["ICRS","Galactic"], "Unrecognized reference system!"
		assert velocity_model in ["joint","constant","linear"],"Unrecognized velocity model!"
		assert sampling_space in ["observed","physical"],"Unrecognized sampling space"

		self.D                = dimension 
		self.zero_points      = zero_points
		self.dir_out          = dir_out
		self.indep_measures   = indep_measures
		
		self.reference_system = reference_system
		self.velocity_model   = velocity_model
		self.sampling_space   = sampling_space

		self.file_ids         = self.dir_out+"/Identifiers.csv"
		self.file_obs         = self.dir_out+"/Observations.nc"
		self.file_chains      = self.dir_out+"/Chains.nc"
		self.file_start       = self.dir_out+"/Initialization.pkl"
		self.file_prior       = self.dir_out+"/Prior.nc"

		self.mas2deg  = 1.0/(60.*60.*1000.)

		self.idx_pma    = 3
		self.idx_pmd    = 4
		self.idx_plx    = 2

		if self.D == 1:
			index_obs   = [0,1,2,8]
			index_mu    = [2]
			index_sd    = [8]
			index_corr  = []
			self.idx_plx = 0
			index_nan   = index_obs.copy()

		elif self.D == 3:
			index_obs  = [0,1,2,6,7,8,12,13,16]
			index_mu   = [0,1,2]
			index_sd   = [6,7,8]
			index_corr = [12,13,16]
			index_nan  = index_obs.copy()

		elif self.D == 6:
			index_obs  = list(range(22))
			index_mu   = [0,1,2,3,4,5]
			index_sd   = [6,7,8,9,10,11]
			index_corr = [12,13,14,15,16,17,18,19,20,21]
			idx_plx    = 2
			#---- Allow missing in radial_velocity ----
			index_nan  = index_obs.copy()
			index_nan.remove(5)
			index_nan.remove(11)
			#-----------------------------------------

		else:
			sys.exit("Dimension not valid!")

		self.names_obs  = [gaia_observables[i] for i in index_obs]
		self.names_mu   = [gaia_observables[i] for i in index_mu]
		self.names_sd   = [gaia_observables[i] for i in index_sd]
		self.names_corr = [gaia_observables[i] for i in index_corr]
		self.names_nan  = [gaia_observables[i] for i in index_nan]
		self.names_coords = coordinates[:dimension]

		self.id_name = id_name
		self.gaia_observables = sum([[id_name],gaia_observables],[]) 

		#============= Transformations ====================================
		if reference_system == "ICRS":
			if self.D == 3:
				self.forward  = icrs_xyz_to_radecplx
				self.backward = np_radecplx_to_icrs_xyz

			elif self.D == 6:
				self.forward  = icrs_xyzuvw_to_astrometry_and_rv
				self.backward = np_astrometry_and_rv_to_icrs_xyzuvw

			else:
				if self.sampling_space == "physical":
					self.forward  = pc2mas
					self.backward = mas2pc
				else:
					self.forward  = Iden
					self.backward = Iden
			
		elif reference_system == "Galactic":
			if self.D == 3:
				self.forward  = galactic_xyz_to_radecplx
				self.backward = np_radecplx_to_galactic_xyz

			elif self.D == 6:
				self.forward  = galactic_xyzuvw_to_astrometry_and_rv
				self.backward = np_astrometry_and_rv_to_galactic_xyzuvw

			else:
				if self.sampling_space == "physical":
					self.forward  = pc2mas
					self.backward = mas2pc
				else:
					self.forward  = Iden
					self.backward = Iden
		else:
			sys.exit("Reference system not accepted")
		#==================================================================

		if self.D in [3,6]:
			assert self.sampling_space == "physical", "3D and 6D models work only in the physical space."



	def load_data(self,file_data,
		corr_func="Lindegren+2020",
		sky_error_factor=1.e6,
		*args,
		**kwargs):
		"""
		This function reads the data.

		Arguments:
		file_data (string): The path to a CSV file.

		corr_func (string): Type of angular correlation.

		Other arguments are passed to pandas.read_csv function

		"""

		#------- Reads the data ---------------------------------------------------
		data  = pn.read_csv(file_data,usecols=self.gaia_observables,*args,**kwargs) 

		#---------- Order ----------------------------------
		data  = data.reindex(columns=self.gaia_observables)

		#------- ID as string ----------------------------
		data[self.id_name] = data[self.id_name].astype('str')

		#----- ID as index ----------------------
		data.set_index(self.id_name,inplace=True)

		#-------- Drop NaNs ----------------------------
		data.dropna(subset=self.names_nan,inplace=True,
						thresh=len(self.names_nan))
		#----------------------------------------------

		#--- Sky uncertainty from mas to degrees ------
		data["ra_error"]  *= self.mas2deg
		data["dec_error"] *= self.mas2deg
		#------------------------------------------

		#--- Increase uncertainty ------------------------
		data["ra_error"]  *= sky_error_factor
		data["dec_error"] *= sky_error_factor
		#-------------------------------------------------

		#---------- Zero-points --------------------
		for key,val in self.zero_points.items():
			data[key] -= val
		#-------------------------------------------

		#--------- Mean values -----------------------------
		mean_observed = data[self.names_mu].mean()
		if "ra" in self.names_mu:
			mean_observed["ra"] = circmean(
				np.array(data["ra"])*u.deg).to(u.deg).value
		self.mean_observed = mean_observed.values
		#---------------------------------------------------

		#------------- Observed --------------------------------
		observed = data[self.names_mu].copy()
		if self.D == 6:
			observed.fillna(value={
				"radial_velocity":mean_observed["radial_velocity"]},
				inplace=True)
		observed["parallax"] = observed["parallax"].clip(1e-3,np.inf)
		self.observed = observed.to_numpy()
		#-------------------------------------------------------

		#----- Track ID -------------
		self.ID = data.index.values
		#----------------------------

		self.n_sources,D = np.shape(data)
		if D != 2 :
			RuntimeError("Data have incorrect shape!")

		#==================== Set Mu and Sigma =========================================
		mu_data = np.zeros(self.n_sources*self.D)
		sg_data = np.zeros((self.n_sources*self.D,self.n_sources*self.D))
		idx_tru = np.triu_indices(self.D,k=1)
		if self.D == 6:
			#----- There is no correlation with r_vel ---
			idi = np.where(idx_tru[1] != 5)[0]
			idx_tru = (idx_tru[0][idi],idx_tru[1][idi])

		for i,(ID,datum) in enumerate(data.iterrows()):
			#--------------------------
			ida  = range(i*self.D,i*self.D + self.D)
			mu   = np.array(datum[self.names_mu])
			sd   = np.array(datum[self.names_sd])
			corr = np.array(datum[self.names_corr])

			#-------- Correlation matrix of uncertainties ---------------
			rho  = np.zeros((self.D,self.D))
			rho[idx_tru] = corr
			rho  = rho + rho.T + np.eye(self.D)

			#-------- Covariance matrix of uncertainties ----------------------
			sigma = np.diag(sd).dot(rho.dot(np.diag(sd)))
			
			#---------- Insert source data --------------
			mu_data[ida] = mu
			sg_data[np.ix_(ida,ida)] = sigma
		#=========================================================================

		#----- Save identifiers --------------------------
		df = pn.DataFrame(self.ID,columns=[self.id_name])
		df.to_csv(path_or_buf=self.file_ids,index=False)
		#------------------------------------------------

		#-------- Observations to InferenceData ---------------
		df = pn.DataFrame(mu_data,
			columns=["obs"],
			index=pn.MultiIndex.from_product(
			iterables=[[0],[0],self.ID,self.names_mu],
			names=['chain', 'draw','source_id','observable']))
		xdata = xarray.Dataset.from_dataframe(df)
		observed = az.InferenceData(observed_data=xdata)
		az.to_netcdf(observed,self.file_obs)
		#------------------------------------------------------

		#===================== Set correlations amongst stars ===========================
		if not self.indep_measures :
			print("Using {} spatial correlation function".format(corr_func))
			#------ Obtain array of positions ------------
			positions = data[["ra","dec"]].to_numpy()

			#------ Angular separations ----------
			theta = AngularSeparation(positions)

			#------ Covariance in parallax -----
			cov_plx = CovarianceParallax(theta,case=corr_func)

			#-------- Test positive definiteness ------------------------------------------------
			try:
				np.linalg.cholesky(cov_plx)
			except np.linalg.LinAlgError as e:
				sys.exit("Covariance matrix of parallax correlations is not positive definite!")
			#------------------------------------------------------------------------------------

			#------ Add parallax covariance -----------------------
			ida_plx = [i*self.D + self.idx_plx for i in range(self.n_sources)]
			sg_data[np.ix_(ida_plx,ida_plx)] += cov_plx
			#------------------------------------------------------
			
			if self.D == 6:
				#------ Covariance in PM ----------------------------
				# Same for mu_alpha and mu_delta
				cov_pms = CovariancePM(theta,case=corr_func)

				#-------- Test positive definiteness ------------------------------------------------
				try:
					np.linalg.cholesky(cov_pms)
				except np.linalg.LinAlgError as e:
					sys.exit("Covariance matrix of proper motions correlations is not positive definite!")
				#------------------------------------------------------------------------------------

				#------ Add PM covariances -----------------------
				ida_pma = [i*self.D + self.idx_pma for i in range(self.n_sources)]
				ida_pmd = [i*self.D + self.idx_pmd for i in range(self.n_sources)]

				sg_data[np.ix_(ida_pma,ida_pma)] += cov_pms
				sg_data[np.ix_(ida_pmd,ida_pmd)] += cov_pms

		#------------ Project into observed subspace -------------
		idx_obs = np.where(np.isfinite(mu_data))[0]
		mu_data = mu_data[idx_obs]
		sg_data = sg_data[np.ix_(idx_obs,idx_obs)]
		#-------------------------------------------------------

		#------- Compute sd or cholesky -----------
		if self.indep_measures:
			tau_data = np.sqrt(np.diag(sg_data))
		else:
			tau_data = np.linalg.cholesky(sg_data)
		#------------------------------------------

		#---- Save as class variables -------
		self.idx_data  = idx_obs
		self.mu_data  = mu_data
		self.sg_data  = sg_data
		self.tau_data = tau_data
		#------------------------------------
		#=================================================================================

		print("Data correctly loaded")


	def setup(self,prior,
				parameters,
				hyper_parameters,
				parametrization,
				):
		'''
		Set-up the model with the corresponding dimensions and data
		'''

		self.prior            = prior
		self.parameters       = parameters.copy()
		self.hyper            = hyper_parameters.copy()
		self.parametrization  = parametrization
		

		print(15*"+", " Prior setup ", 15*"+")
		print("Type of prior: ",self.prior)
		print("Working in the {} reference system".format(self.reference_system))

		msg_alpha = "hyper_alpha must be specified."
		msg_beta  = "hyper_beta must be specified."
		msg_gamma = "hyper_gamma must be specified."
		msg_delta = "hyper_delta must be specified."
		msg_nu    = "hyper_nu must be specified."
		msg_central = "Only the central parametrization is valid for this configuration."
		msg_non_central = "Only the non-central parametrization is valid for this configuration."
		msg_weights = "weights must be greater than 5%."

		assert self.parametrization in ["central","non-central"], "Error in parametrization"

		#============== Mixtures =====================================================
		if "GMM" in self.prior:
			n_components = self.hyper["n_components"]
			names_components = list(string.ascii_uppercase)[:n_components]

			if self.parameters["weights"] is None:
				assert self.hyper["delta"] is not None, msg_delta
			else:
				#-------------- Read from input file ----------------------
				if isinstance(self.parameters["weights"],str):
					#---- Extract scale parameters ------------
					wgh = pn.read_csv(self.parameters["weights"],
								usecols=["Parameter","MAP"])
					wgh = wgh[wgh["Parameter"].str.contains("weights")]
					#------------------------------------------

					#---- Set weights ----------------------------
					self.parameters["weights"] = wgh["MAP"].values
					#-----------------------------------------------
				#-----------------------------------------------------------

				#--------- Verify weights ---------------------------------
				print("The weights parameter is fixed to:")
				print(self.parameters["weights"])
				assert len(self.parameters["weights"]) == n_components, \
					"The size of the weights parameter is incorrect!"
				#assert np.min(self.parameters["weights"])> 0.05, msg_weights
				#-----------------------------------------------------------

			assert self.parametrization == "central", msg_central
		#============================================================================

		#====================== Location ==========================================================
		if self.parameters["location"] is None:
			#---------------- Alpha ---------------------------------------------------------
			print("The alpha hyper-parameter has been set to:")
			if self.hyper["alpha"] is None:
				#-- Cluster dispersion ----
				uvw_sd = 5.
				xyz_fc = 0.2
				#-------------------------

				if self.D == 1:
					#---------- Mean distance ------------
					d = self.backward(self.mean_observed[0])
					#------------------------------------

					#---------- Dispersion -----------------
					d_sd = xyz_fc*d
					#---------------------------------------

					self.hyper["alpha"] = {
						"loc":[d],
						"scl":[d_sd]}
				
				elif self.D == 3:
					#------------ Cluster mean coordinates -------------------
					x,y,z = self.backward(self.mean_observed[np.newaxis,:]).flatten()
					#----------------------------------------------------------

					#---------- Dispersion -----------------
					xyz_sd = xyz_fc*np.sqrt(x**2 + y**2 + z**2)
					#---------------------------------------

					self.hyper["alpha"] = {
						"loc":[x,y,z],
						"scl":[xyz_sd,xyz_sd,xyz_sd]}

				elif self.D == 6:
					#------------ Cluster mean coordinates -------------------
					x,y,z,u,v,w = self.backward(self.mean_observed[np.newaxis,:]).flatten()
					#----------------------------------------------------------

					#---------- Dispersion -----------------
					xyz_sd = xyz_fc*np.sqrt(x**2 + y**2 + z**2)
					#---------------------------------------

					self.hyper["alpha"] = {
					"loc":[x,y,z,u,v,w],
					"scl":[xyz_sd,xyz_sd,xyz_sd,uvw_sd,uvw_sd,uvw_sd]}

			for name,loc,scl,unit in zip(
				self.names_coords,
				self.hyper["alpha"]["loc"],
				self.hyper["alpha"]["scl"],
				np.array(["pc","pc","pc","km/s","km/s","km/s"])[:self.D]):
				print("{0}: {1:2.1f} +/- {2:2.1f} {3}".format(name,loc,scl,unit))	
			#---------------------------------------------------------------------------------

		else:
			#-------------------------- Fixed value -----------------------------------------------------------
			msg_loc = "ERROR: The size of the provided location parameter is incorrect!"
			msg_gmm = "ERROR: The list type for fixing the location parameter is only valid for mixture models"

			#-------------- Read from input file ----------------------------------
			if isinstance(self.parameters["location"],str):
				#---- Extract parameters ------------------------
				pars = pn.read_csv(self.parameters["location"],
							usecols=["Parameter","MAP"])
				#------------------------------------------------

				
				#------------------- Extraction --------------------------
				if "GMM" in self.prior:
					locs = []
					for name in names_components:
						selection = pars["Parameter"].str.contains(
									"loc[{0}".format(name),regex=False)
						loc = pars.loc[selection,"MAP"].values
						assert loc.shape[0] == self.D, msg_loc
						locs.append(loc)

					self.parameters["location"] = locs

				else:
					mask_loc = pars["Parameter"].str.contains("loc")
					loc = pars.loc[mask_loc,"MAP"]

					self.parameters["location"] = np.array(loc.values)
				#----------------------------------------------------------

			print("The location parameter has been fixed to:")

			if isinstance(self.parameters["location"],float):
				assert self.D == 1,"ERROR: float type in location parameter is only valid in 1D"
				self.parameters["location"] = np.array([self.parameters["location"]])
			
			if isinstance(self.parameters["location"],np.ndarray):
				assert self.parameters["location"].shape[0] == self.D, msg_loc
				for name,loc in zip(self.names_coords,self.parameters["location"]):
					unit = "pc" if name in ["X","Y","Z"] else "km.s-1"
					print("{0}: {1:2.1f} [{2}]".format(name,loc,unit))

			#-------------- Mixture model ----------------------------
			elif isinstance(self.parameters["location"],list):
				assert "GMM" in self.prior, msg_gmm
				for name,loc in zip(names_components,self.parameters["location"]):
					print("Component {0}".format(name))
					assert loc.shape[0] == self.D, msg_loc
					for name,loc in zip(self.names_coords,loc):
						print("{0}: {1:2.1f} pc".format(name,loc))		
			#----------------------------------------------------------

			else:
				sys.exit("ERROR: Unrecognized type of location parameter!"\
					"Must be None, string, float (1D), numpy array or list of numpy arrays (mixture models).")
		#-----------------------------------------------------------------------
		#==============================================================================================
		
		#============================= Scale ===========================================================
		if self.parameters["scale"] is None:
			if self.hyper["beta"] is None:
				self.hyper["beta"] = np.array([10.0,10.0,10.0,2.0,2.0,2.0])[:self.D]
			else:
				if isinstance(self.hyper["beta"],float):
					assert self.D == 1,\
						"ERROR: float type in beta hyper-parameter is only valid in 1D"
					self.hyper["beta"] = np.array([self.hyper["beta"]])
				elif isinstance(self.hyper["beta"],list):
					assert len(self.hyper["beta"]) == self.D,\
						"ERROR: incorrect length of hyperparameter beta!"
					self.hyper["beta"] = np.array(self.hyper["beta"])
				elif isinstance(self.hyper["beta"],np.ndarray):
					assert self.hyper["beta"].ndim == 1 and self.hyper["beta"].shape[0] == self.D,\
						"ERROR: incorrect shape of hyperparameter beta!"
				else:
					sys.exit("ERROR:Unrecognized type of hyper_beta")

			print("The beta hyper-parameter has been set to:")
			for name,scl,unit in zip(
				self.names_coords,
				self.hyper["beta"],
				np.array(["pc","pc","pc","km/s","km/s","km/s"])[:self.D]
				):
				print("{0}: {1:2.1f} {2}".format(name,scl,unit))	

		else:

			#-------------------------- Fixed value -----------------------------------------------------------
			msg_scl = "ERROR: The size of the provided scale parameter is incorrect!"
			msg_gmm = "ERROR: The list type for fixing the scale parameter is only valid for mixture models"

			if isinstance(self.parameters["scale"],str):
				#---- Extract parameters ------------
				pars = pn.read_csv(self.parameters["scale"],
							usecols=["Parameter","MAP"])
				pars.fillna(value=1.0,inplace=True)
				#------------------------------------------

				#-------- Extraction is prior dependent ----------------
				if "GMM" in self.prior:
					stds = []
					cors = []
					covs = []
					for name in names_components:
						#------------- Stds ---------------------------
						mask_std = pars["Parameter"].str.contains(
									"std[{0}".format(name),regex=False)
						std = pars.loc[mask_std,"MAP"].values
						#------------------------------------------------

						stds.append(std)

						if self.D != 1:
							#----------- Correlations ------------------------
							mask_cor = pars["Parameter"].str.contains(
										"corr[{0}".format(name),regex=False)
							cor = pars.loc[mask_cor,"MAP"].values
							#--------------------------------------------------

							#---- Construct covariance --------------
							std = np.diag(std)
							cor = np.reshape(cor,(self.D,self.D))
							cov = np.dot(std,cor.dot(std))
							#-----------------------------------------

							#--- Append -------
							cors.append(cor)
							covs.append(cov)
							#------------------
			
					if self.D == 1:
						self.parameters["scale"] = stds
					else:
						self.parameters["scale"] = covs

				else:
					#--------- Standard deviations --------------------
					mask_stds = pars["Parameter"].str.contains("std")
					stds = pars.loc[mask_stds,"MAP"].values
					#--------------------------------------------------

					if self.D == 1:
						self.parameters["scale"] = np.array(stds)
					else:
						#----------- Correlations -----------------------
						mask_corr = pars["Parameter"].str.contains('corr')
						corr = pars.loc[mask_corr,"MAP"].values
						#------------------------------------------------

						#---- Construct covariance --------------
						stds = np.diag(stds)
						corr = np.reshape(corr,(self.D,self.D))
						cov = np.dot(stds,corr.dot(stds))
						#-----------------------------------------

						self.parameters["scale"] = np.array(cov)

					
			#---------------------------------------------------------------------

			print("The scale parameter has been fixed to:")

			if isinstance(self.parameters["scale"],float):
				assert self.D == 1,"ERROR: float type in scale parameter is only valid in 1D"
				self.parameters["scale"] = np.array([self.parameters["scale"]])

			if isinstance(self.parameters["scale"],np.ndarray):
				if self.D == 1:
					assert self.parameters["scale"].shape[0] == 1, msg_scl
				else:
					assert self.parameters["scale"].shape == (self.D,self.D), msg_scl

				print(self.parameters["scale"])

			#-------------- Mixture model ----------------------------
			elif isinstance(self.parameters["scale"],list):
				assert "GMM" in self.prior, msg_gmm
				for name,scl in zip(names_components,self.parameters["scale"]):
					if self.D == 1:
						assert scl.shape[0] == 1, msg_scl
					else:
						assert scl.shape == (self.D,self.D), msg_scl
					print("Component {0}".format(name))
					print(scl)
			#----------------------------------------------------------	
			else:
				sys.exit("ERROR: Unrecognized type of scale parameter!"\
					"Must be None, string, float (1D), numpy array or list of numpy arrays (mixture models).")
		#==============================================================================================

		#=========================== Initial values =======================================
		self.starting_points = {"{0}D::source".format(self.D):self.backward(self.observed)}
		#==================================================================================

		#==================== Miscelaneous ============================================================
		if self.parameters["scale"] is None and self.hyper["eta"] is None:
			self.hyper["eta"] = 1.0
			print("The eta hyper-parameter has been set to:")
			print(self.hyper["eta"])

		if self.prior in ["EDSD","GGD","Uniform","EFF","King"]:
			assert self.D == 1, "{0} prior is only valid for 1D version.".format(self.prior)

		if self.prior == "StudentT":
			assert "nu" in self.hyper, msg_nu
			if self.hyper["nu"] is None:
				self.hyper["nu"] = {"alpha":1.0,"beta":10}
			else:
				assert "alpha" in self.hyper["nu"], "Error in hyper_nu"
				assert "beta" in self.hyper["nu"], "Error in hyper_nu"

			print("The nu hyper parameter has been set to:")
			print("alpha: {0:2.1f}, beta: {1:2.1f}".format(
					self.hyper["nu"]["alpha"],self.hyper["nu"]["beta"]))
		else:
			self.hyper["nu"] = None

		if self.prior == "FGMM":
			assert "field_scale" in self.parameters, "Model FGMM needs the 'field_scale' parameter"
			assert isinstance(self.parameters["field_scale"],list), "Error. The field_scale must be a list of floats!"
			assert len(self.parameters["field_scale"]) == self.D, "Error: the length of field_scale must match model dimension"

		if self.prior in ["King","EFF"]:
			if self.prior == "KING" and self.parameters["rt"] is None:
				assert self.hyper["gamma"] is not None, msg_gamma
			if self.prior == "EFF" and self.parameters["gamma"] is None:
				assert self.hyper["gamma"] is not None, msg_gamma
		else:
			self.hyper["gamma"] = None

		if self.velocity_model in ["constant","linear"]:
			if "age" in self.parameters.keys():
				if self.parameters["age"] is None:
					test_hyper = ( "loc" in self.hyper["age"] and 
								   "scl" in self.hyper["age"] and
								   "upper" in self.hyper["age"])
					assert test_hyper, "Error: The loc, scl, and upper hyper_parameter of age must be set!"
					print("The age prior has been set to:")
					print("age ~ TruncatedNormal(loc={0:2.1f},scale={1:2.1f},lower=0.0,upper={2:2.1f}) [Myr]".format(
						self.hyper['age']['loc'],self.hyper['age']['scl'],self.hyper['age']['upper']))
				elif isinstance(self.parameters["age"],float):
					print("The age parameter has been set to:")
					print(self.parameters["age"])
				else:
					sys.exit("Error: The age parameter must be float or None!")

				if "tau" not in self.hyper.keys():
					self.hyper["tau"]={"d":3.,"p":2.}
					print("The tau hyper_parameter has been set to:")
					print(self.hyper["tau"])
				assert "d" in self.hyper["tau"] and "p" in self.hyper["tau"],"Error: d and p must be in the tau hyper_parameter!"
				assert self.hyper["tau"]["d"] is None or isinstance(self.hyper["tau"]["d"],float),"Error: the d hyper_parameter must be None or float!"
				assert self.hyper["tau"]["p"] is None or isinstance(self.hyper["tau"]["p"],float),"Error: the p hyper_parameter must be None or float!"
				if self.hyper["tau"]["d"] is None:
					if "hyper_tau_d_beta" not in self.hyper["tau"]:
						self.hyper["tau"]["hyper_tau_d_beta"] = 1./3.0
					assert isinstance(self.hyper["tau"]["hyper_tau_d_beta"],float),"Error: the hyper_tau_d_beta must be float!"
					print("The tau_d prior has been set to:")
					print("tau_d ~ Gamma(alpha=2,beta={0:2.2f})".format(self.hyper["tau"]["hyper_tau_d_beta"]))

				if self.hyper["tau"]["p"] is None:
					if "hyper_tau_p_beta" not in self.hyper["tau"]:
						self.hyper["tau"]["hyper_tau_p_beta"] = 1./2.0
					assert isinstance(self.hyper["tau"]["hyper_tau_p_beta"],float),"Error: the hyper_tau_p_beta must be float!"
					print("The tau_p prior has been set to:")
					print("tau_p ~ Gamma(alpha=2,beta={0:2.2f})".format(self.hyper["tau"]["hyper_tau_p_beta"]))
			else:
				if self.parameters["kappa"] is None :
					print("The kappa prior has been set to:")
					if self.hyper["kappa"] is None:
						self.hyper["kappa"]={"loc":0.0,"scl":0.1}
						print("kappa ~ Normal(loc={0:2.3f},scl={1:2.3f}) [km.s-1.pc-1]".format(self.hyper["kappa"]["loc"],self.hyper["kappa"]["scl"]))
				elif isinstance(self.parameters["kappa"],np.ndarray):
					print("The kappa parameter has been fixed to:")
					print(self.parameters["kappa"])
				else:
					sys.exit("Error: The kappa parameter must be None or ndarray.shape == (3)")

			if self.velocity_model == "linear":
				if self.parameters["omega"] is None :
					print("The omega prior has been set to:")
					if self.hyper["omega"] is None:
						self.hyper["omega"] = 0.1
					print("omega~Normal(loc=0.000,scl={0:2.3f}) [km.s-1.pc-1]".format(self.hyper["omega"]))
				else:
					print("The omega parameter has been fixed to:")
					print(self.parameters["omega"])
		#===========================================================================================================

		if self.D == 1:
			self.Model = Model1D(
								n_sources=self.n_sources,
								mu_data=self.mu_data,
								tau_data=self.tau_data,
								indep_measures=self.indep_measures,
								dimension=self.D,
								prior=self.prior,
								parameters=self.parameters,
								hyper_alpha=self.hyper["alpha"],
								hyper_beta=self.hyper["beta"],
								hyper_gamma=self.hyper["gamma"],
								hyper_delta=self.hyper["delta"],
								hyper_nu=self.hyper["nu"],
								transformation=self.forward,
								parametrization=self.parametrization,
								identifiers=self.ID,
								coordinates=self.names_coords,
								observables=self.names_mu)

		elif self.D in [3,6] and self.velocity_model == "joint":
			self.Model = Model3D6D(
								n_sources=self.n_sources,
								mu_data=self.mu_data,
								tau_data=self.tau_data,
								idx_data=self.idx_data,
								indep_measures=self.indep_measures,
								dimension=self.D,
								prior=self.prior,
								parameters=self.parameters,
								hyper_alpha=self.hyper["alpha"],
								hyper_beta=self.hyper["beta"],
								hyper_gamma=self.hyper["gamma"],
								hyper_delta=self.hyper["delta"],
								hyper_eta=self.hyper["eta"],
								hyper_nu=self.hyper["nu"],
								transformation=self.forward,
								parametrization=self.parametrization,
								identifiers=self.ID,
								coordinates=self.names_coords,
								observables=self.names_mu)

		elif self.D == 6 and self.velocity_model != "joint":
			self.Model = Model6D_linear(n_sources=self.n_sources,
								mu_data=self.mu_data,
								tau_data=self.tau_data,
								idx_data=self.idx_data,
								indep_measures=self.indep_measures,
								prior=self.prior,
								parameters=self.parameters,
								hyper_alpha=self.hyper["alpha"],
								hyper_beta=self.hyper["beta"],
								hyper_gamma=self.hyper["gamma"],
								hyper_delta=self.hyper["delta"],
								hyper_eta=self.hyper["eta"],
								hyper_kappa=self.hyper["kappa"],
								hyper_omega=self.hyper["omega"],
								hyper_nu=self.hyper["nu"],
								hyper_tau=self.hyper["tau"],
								hyper_age=self.hyper["age"],
								transformation=self.forward,
								parametrization=self.parametrization,
								velocity_model=self.velocity_model,
								identifiers=self.ID,
								coordinates=self.names_coords,
								observables=self.names_mu)
		else:
			sys.exit("Non valid dimension or velocity model!")

		print((30+13)*"+")

	def run(self,
		tuning_iters=2000,
		sample_iters=2000,
		target_accept=0.6,
		chains=2,
		cores=2,
		step=None,
		step_size=None,
		init_method="advi+adapt_diag",
		init_iters=int(1e5),
		init_absolute_tol=5e-3,
		init_relative_tol=1e-5,
		init_plot_iters=int(1e4),
		init_refine=False,
		prior_predictive=False,
		prior_iters=500,
		progressbar=True,
		nuts_sampler="numpyro",
		random_seed=None):
		"""
		Performs the MCMC run.
		Arguments:
		sample_iters (integer):    Number of MCMC iterations.
		tuning_iters (integer):    Number of burning iterations.
		"""

		#------- Step_size ----------
		if step_size is None:
			if self.D == 1:
				step_size = 1.e-1
			elif self.D == 3:
				step_size = 1.e-2
			else:
				step_size = 1.e-3
		#---------------------------

		if not os.path.exists(self.file_chains):
			#================== Optimization =============================================
			if os.path.exists(self.file_start):
				print("Reading initial positions ...")
				in_file = open(self.file_start, "rb")
				approx = dill.load(in_file)
				in_file.close()
				start = approx["initial_points"][0]
			else:
				approx = None
				start = self.starting_points
				print("Finding initial positions ...")

			if approx is None or (approx is not None and init_refine):
				# # -------- Fix problem with initial solution of cholesky cov-packed ----------
				# name_ccp = "_cholesky-cov-packed__" 
				# for key,value in start.copy().items():
				# 	if name_ccp in key:
				# 		del start[key]
				# # TO BE REMOVED once pymc5 solves this issue
				# #----------------------------------------------------------------------------

				random_seed_list = pymc.util._get_seeds_per_chain(random_seed, chains)
				cb = [pm.callbacks.CheckParametersConvergence(
						tolerance=init_absolute_tol, diff="absolute",ord=None),
					  pm.callbacks.CheckParametersConvergence(
						tolerance=init_relative_tol, diff="relative",ord=None)]

				approx = pm.fit(
					start=start,
					random_seed=random_seed_list[0],
					n=init_iters,
					method="advi",
					model=self.Model,
					callbacks=cb,
					progressbar=True,
					#test_optimizer=pm.adagrad#_window
					)

				#------------- Plot Loss ----------------------------------
				plt.figure()
				plt.plot(approx.hist[-init_plot_iters:])
				plt.xlabel("Last {0} iterations".format(init_plot_iters))
				plt.ylabel("Average Loss")
				plt.savefig(self.dir_out+"/Initializations.png")
				plt.close()
				#-----------------------------------------------------------

				approx_sample = approx.sample(
					draws=chains, 
					random_seed=random_seed_list[0],
					return_inferencedata=False
					)

				initial_points = [approx_sample[i] for i in range(chains)]
				sd_point = approx.std.eval()
				mu_point = approx.mean.get_value()
				approx = {
					"initial_points":initial_points,
					"mu_point":mu_point,
					"sd_point":sd_point
					}

				out_file = open(self.file_start, "wb")
				dill.dump(approx, out_file)
				out_file.close()

				#------------------ Save initial point ------------------------------
				df = pn.DataFrame(data=initial_points[0]["{0}D::true".format(self.D)],
					columns=self.names_mu)
				df.to_csv(self.dir_out+"/initial_true.csv",index=False)
				df = pn.DataFrame(data=initial_points[0]["{0}D::source".format(self.D)],
					columns=self.names_coords)
				df.to_csv(self.dir_out+"/initial_source.csv",index=False)
				#---------------------------------------------------------------------

			#----------- Extract ---------------------
			mu_point = approx["mu_point"]
			sd_point = approx["sd_point"]
			initial_points = approx["initial_points"]
			#-----------------------------------------

			# # -------- Fix problem with initial solution of cholesky cov-packed ----------
			# name_ccp = "_cholesky-cov-packed__" 
			# for vals in initial_points:
			# 	for key,value in vals.copy().items():
			# 		if name_ccp in key:
			# 			del vals[key]
			# # TO BE REMOVED once pymc5 solves this issue
			# #----------------------------------------------------------------------------

			#================================================================================

			#=================== Sampling ==================================================
			if nuts_sampler == "pymc":
				#--------------- Prepare step ---------------------------------------------
				# Only valid for nuts_sampler == "pymc". 
				# The other samplers adapt steps independently.
				potential = pymc.step_methods.hmc.quadpotential.QuadPotentialDiagAdapt(
							n=len(mu_point),
							initial_mean=mu_point,
							initial_diag=sd_point**2, 
							initial_weight=10)

				step = pm.NUTS(
						potential=potential,
						model=self.Model,
						target_accept=target_accept
						)
				#----------------------------------------------------------------------------

				print("Sampling the model ...")

				#---------- Posterior -----------
				trace = pm.sample(
					draws=sample_iters,
					initvals=initial_points,
					step=step,
					nuts_sampler=nuts_sampler,
					tune=tuning_iters,
					chains=chains, 
					cores=cores,
					progressbar=progressbar,
					discard_tuned_samples=True,
					return_inferencedata=True,
					nuts_sampler_kwargs={"step_size":step_size},
					model=self.Model
					)
				#--------------------------------
			else:
				#---------- Posterior -----------
				trace = pm.sample(
					draws=sample_iters,
					initvals=initial_points,
					nuts_sampler=nuts_sampler,
					tune=tuning_iters,
					chains=chains, 
					cores=cores,
					progressbar=progressbar,
					target_accept=target_accept,
					discard_tuned_samples=True,
					return_inferencedata=True,
					#nuts_sampler_kwargs={"step_size":step_size},
					model=self.Model
					)
				#--------------------------------

			#--------- Save with arviz ------------
			print("Saving posterior samples ...")
			az.to_netcdf(trace,self.file_chains)
			#-------------------------------------
			del trace
			#================================================================================

		
		if prior_predictive and not os.path.exists(self.file_prior):
				#-------- Prior predictive -------------------
				print("Sampling prior predictive ...")
				prior_pred = pm.sample_prior_predictive(
							samples=prior_iters,
							model=self.Model)
				print("Saving prior predictive ...")
				az.to_netcdf(prior_pred,self.file_prior)
				#---------------------------------------------

		print("Sampling done!")
		


	def load_trace(self,file_chains=None):
		'''
		Loads a previously saved sampling of the model
		'''

		file_chains = self.file_chains if (file_chains is None) else file_chains

		if not hasattr(self,"ID"):
			#----- Load identifiers ------
			self.ID = pn.read_csv(self.file_ids).to_numpy().flatten()

		print("Loading existing samples ... ")

		#---------Load posterior ---------------------------------------------------
		try:
			posterior = az.from_netcdf(file_chains)
		except ValueError:
			sys.exit("ERROR at loading {0}".format(file_chains))
		#------------------------------------------------------------------------

		#----------- Load prior -------------------------------------------------
		try:
			prior = az.from_netcdf(self.file_prior)
		except:
			prior = None
			self.ds_prior = None
		
		if prior is not None:
			posterior.extend(prior)
			self.ds_prior = posterior.prior
		#-------------------------------------------------------------------------

		self.trace = posterior

		#---------Load posterior ---------------------------------------------------
		try:
			self.ds_posterior = self.trace.posterior
		except ValueError:
			sys.exit("There is no posterior in trace")
		#------------------------------------------------------------------------

		#------- Variable names -----------------------------------------------------------
		source_variables = list(filter(lambda x: "source" in x, self.ds_posterior.data_vars))
		cluster_variables = list(filter(lambda x: ( ("loc" in x) 
											or ("corr" in x)
											or ("std" in x)
											or ("std" in x)
											or ("weights" in x)
											or ("beta" in x)
											or ("gamma" in x)
											or ("rt" in x)
											or ("kappa" in x)
											or ("omega" in x)
											or ("nu" in x)
											or ("tau" in x)
											or ("age" in x)
											),
											self.ds_posterior.data_vars))
	
		trace_variables = cluster_variables.copy()
		stats_variables = cluster_variables.copy()
		tensor_variables= cluster_variables.copy()
		cluster_loc_var = cluster_variables.copy()
		cluster_std_var = cluster_variables.copy()
		cluster_cor_var = cluster_variables.copy()
		cluster_ppc_var = cluster_variables.copy()

		#----------- Case specific variables -------------
		tmp_srces = source_variables.copy()
		tmp_plots = cluster_variables.copy()
		tmp_stats = cluster_variables.copy()
		tmp_loc   = cluster_variables.copy()
		tmp_stds  = cluster_variables.copy()
		tmp_corr  = cluster_variables.copy()
		tmp_ppc   = cluster_variables.copy()

		for var in tmp_srces:
			if "_pos" in var or "_vel" in var:
				source_variables.remove(var)

		for var in tmp_plots:
			if self.D in [3,6]:
				if "corr" in var:
					trace_variables.remove(var)
				if "lnv" in var and "std" not in var:
					trace_variables.remove(var)

		for var in tmp_stats:
			if self.D in [3,6]:
				if not ("loc" in var 
					or "std" in var
					or "weights" in var
					or "nu" in var
					or "corr" in var 
					or "omega" in var
					or "kappa" in var
					or "tau" in var
					or "age" in var
					):
					stats_variables.remove(var)

		for var in tmp_loc:
			if "loc" not in var:
				cluster_loc_var.remove(var)

		for var in tmp_stds:
			if "std" not in var:
				cluster_std_var.remove(var)

		for var in tmp_corr:
			if "corr" not in var:
				cluster_cor_var.remove(var)

		for var in tmp_ppc:
			if "corr" in var:
				cluster_ppc_var.remove(var)

		#----------------------------------------------------

		self.source_variables  = source_variables
		self.cluster_variables = cluster_variables
		self.trace_variables   = trace_variables
		self.stats_variables   = stats_variables
		self.loc_variables     = cluster_loc_var
		self.std_variables     = cluster_std_var
		self.cor_variables     = cluster_cor_var
		self.chk_variables     = cluster_ppc_var

		# print(self.source_variables)
		# print(self.cluster_variables)
		# print(self.trace_variables )
		# print(self.stats_variables  )
		# print(self.loc_variables    )
		# print(self.std_variables     )
		# print(self.cor_variables     )
		# print(self.chk_variables)
		# sys.exit()

	def convergence(self):
		"""
		Analyse the chains.		
		"""
		print("Computing convergence statistics ...")
		rhat  = az.rhat(self.ds_posterior)
		ess   = az.ess(self.ds_posterior)

		print("Gelman-Rubin statistics:")
		for var in self.ds_posterior.data_vars:
			print("{0} : {1:2.4f}".format(var,np.mean(rhat[var].values)))

		print("Effective sample size:")
		for var in self.ds_posterior.data_vars:
			print("{0} : {1:2.4f}".format(var,np.mean(ess[var].values)))

		# print("Step size:")
		# for i,val in enumerate(self.trace.sample_stats["step_size"].mean(dim="draw")):
		# 	print("Chain {0}: {1:3.8f}".format(i,val))

	def plot_chains(self,
		file_plots=None,
		IDs=None,
		divergences='bottom', 
		figsize=None, 
		lines=None, 
		combined=False,
		compact=False,
		plot_kwargs=None, 
		hist_kwargs=None, 
		trace_kwargs=None,
		fontsize_title=16):
		"""
		This function plots the trace. Parameters are the same as in pymc3
		"""
		if IDs is None and len(self.cluster_variables) == 0:
			return

		print("Plotting traces ...")

		file_plots = self.dir_out+"/Traces.pdf" if (file_plots is None) else file_plots


		pdf = PdfPages(filename=file_plots)

		if IDs is not None:
			#--------- Loop over ID in list ---------------
			for i,ID in enumerate(IDs):
				id_in_IDs = np.isin(self.ID,ID)
				if not np.any(id_in_IDs) :
					sys.exit("{0} {1} is not valid. Use strings".format(self.id_name,ID))
				idx = np.where(id_in_IDs)[0]
				coords = {"source_id":ID}
				plt.figure(0)
				axes = az.plot_trace(self.ds_posterior,
						var_names=self.source_variables,
						coords=coords,
						figsize=figsize,
						lines=lines, 
						combined=combined,
						compact=compact,
						plot_kwargs=plot_kwargs, 
						hist_kwargs=hist_kwargs, 
						trace_kwargs=trace_kwargs)

				for ax in axes:

					# --- Set units in parameters ------------------------------
					if self.transformation == "pc":
						ax[0].set_xlabel("pc")
					else:
						ax[0].set_xlabel("mas")
					#-----------------------------------------------------------

					ax[1].set_xlabel("Iterations")
					ax[0].set_title(None)
					ax[1].set_title(None)
				plt.gcf().suptitle(self.id_name +" "+ID,fontsize=fontsize_title)

					
				#-------------- Save fig --------------------------
				pdf.savefig(bbox_inches='tight')
				plt.close(0)

		for var_name in self.trace_variables:
			axes = az.plot_trace(self.ds_posterior,
					var_names=var_name,
					figsize=figsize,
					lines=lines, 
					combined=combined,
					compact=compact,
					plot_kwargs=plot_kwargs, 
					hist_kwargs=hist_kwargs, 
					trace_kwargs=trace_kwargs,
					labeller=az.labels.NoVarLabeller())

			for ax in axes:
				# --- Set units in parameters ------------------------------
				title = ax[0].get_title()
				if self.sampling_space == "physical":
					if title in ["X","Y","Z"]:
						ax[0].set_xlabel("$pc$")
					if title in ["U","V","W"]:
						ax[0].set_xlabel("$km\\,s^{-1}$")
				else:
					ax[0].set_xlabel("mas")
				if "kappa" in title or "omega" in title:
					ax[0].set_xlabel("$km\\,s^{-1}\\, pc^{-1}$")
					#-----------------------------------------------------------
				ax[1].set_xlabel("Iteration")

			plt.gcf().suptitle("Population parameter: {}".format(var_name),
						fontsize=fontsize_title)
			plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,hspace=0.5,wspace=0.1)
				
			#-------------- Save fig --------------------------
			pdf.savefig(bbox_inches='tight')
			plt.close(1)

		pdf.close()

	def plot_prior_check(self,
		file_plots=None,
		figsize=None,
		):
		"""
		This function plots the prior and posterior distributions.
		"""

		print("Plotting checks ...")
		file_plots = self.dir_out+"/Prior_check.pdf" if (file_plots is None) else file_plots

		pdf = PdfPages(filename=file_plots)
		for var in self.chk_variables:
			plt.figure(0,figsize=figsize)
			az.plot_dist_comparison(self.trace,var_names=var)
			pdf.savefig(bbox_inches='tight')
			plt.close(0)
		pdf.close()

	def _extract(self,group="posterior",n_samples=None,chain=None):
		if group == "posterior":
			data = self.ds_posterior.data_vars
		elif group == "prior":
			data = self.ds_prior.data_vars
		else:
			sys.exit("Group not recognized")

		#================ Sources ============================================
		#------------ Extract sources ---------------------------------------
		srcs = np.array([data[var].values for var in self.source_variables])
		#--------------------------------------------------------------------

		#------ Organize sources -----
		srcs = srcs.squeeze(axis=0)
		srcs = np.moveaxis(srcs,2,0)
		#-----------------------------

		#------ Dimensions -----
		n,nc,ns,nd = srcs.shape
		#-----------------------

		#--- One or multiple chains -----------
		if chain is None:
			#-------- Merge chains ----------
			srcs = srcs.reshape((n,nc*ns,nd))
			#--------------------------------

		else:
			#--- Extract chain --------------------
			srcs = srcs[:,chain].reshape((n,ns,nd))
			#--------------------------------------
		#--------------------------------------

		#------- Sample --------------------
		if n_samples is not None:
			idx = np.random.choice(
				  np.arange(srcs.shape[1]),
						replace=False,
						size=n_samples)
			srcs = srcs[:,idx]
		#------------------------------------
		#=====================================================================


		#============== Parameters ===========================================
		#----------- Extract location ---------------------------------------
		if len(self.loc_variables) == 0:
			locs = np.array(self.parameters["location"])
			locs = np.swapaxes(locs,0,1)[:,np.newaxis,np.newaxis,:]
			locs = np.tile(locs,(1,nc,ns,1))

		else:
			locs = np.array([data[var].values for var in self.loc_variables])
		#--------------------------------------------------------------------

		#----------- Extract stds -------------------------------------------
		if len(self.std_variables) == 0:
			stds = np.array(self.parameters["std"])
			stds = stds[:,np.newaxis,np.newaxis,:]
			stds = np.tile(stds,(1,nc,ns,1))
		else:
			if any(["pos" in var or "vel" in var for var in self.std_variables]):
				for var in self.std_variables:
					if "pos" in var:
						stds_pos = np.array([data[var].values])
					if "vel" in var:
						stds_vel = np.array([data[var].values])
				stds = np.zeros((
							stds_pos.shape[0],
							stds_pos.shape[1],
							stds_pos.shape[2],6))
				stds[:,:,:,:3] = stds_pos
				stds[:,:,:,3:] = stds_vel
			else:
				stds = np.array([data[var].values for var in self.std_variables])
		#------------------------------------------------------------------------

		#----------- Extract correlations -------------------------------
		if len(self.std_variables) == 0:
			cors = np.array(self.parameters["corr"])
			cors = cors[:,np.newaxis,np.newaxis,:]
			cors = np.tile(cors,(1,nc,ns,1,1))
		else:
			if any(["pos" in var or "vel" in var for var in self.cor_variables]):
				for var in self.cor_variables:
					if "pos" in var:
						cors_pos = np.array([data[var].values])
					elif "vel" in var:
						cors_vel = np.array([data[var].values])
					else:
						sys.exit("Neither pos nor vel in variables")
				cors = np.zeros((
							cors_pos.shape[0],
							cors_pos.shape[1],
							cors_pos.shape[2],6,6))
				cors[:,:,:,:3,:3] = cors_pos
				cors[:,:,:,3:,3:] = cors_vel
			elif self.D == 1:
				cors = np.ones_like(stds)
			else:
				cors = np.array([data[var].values for var in self.cor_variables])
		#------------------------------------------------------------------------

		#--------- Reorder indices ----------------------
		if "GMM" in self.prior:
			locs = np.squeeze(locs, axis=0)
			stds = np.squeeze(stds, axis=0)
			cors = np.squeeze(cors, axis=0)

			locs = np.moveaxis(locs,2,0)
			stds = np.moveaxis(stds,2,0)
			cors = np.moveaxis(cors,2,0)

			if str(self.D)+"D::weights" in self.cluster_variables:
				amps = np.array(data[str(self.D)+"D::weights"].values)
				amps = np.moveaxis(amps,2,0)
			else:
				amps = np.array(self.parameters["weights"])
				amps = amps[:,np.newaxis,np.newaxis]
				amps = np.tile(amps,(1,nc,ns))
		else:
			amps = np.ones_like(locs)[:,:,:,0]
		#-------------------------------------------------
		
		#---------- One or multiple chains -------
		ng,nc,ns,nd = locs.shape
		if chain is None:
			#-------- Merge chains --------------
			amps = amps.reshape((ng,nc*ns))
			locs = locs.reshape((ng,nc*ns,nd))
			stds = stds.reshape((ng,nc*ns,nd))
			cors = cors.reshape((ng,nc*ns,nd,nd))
			#------------------------------------

		else:
			#--- Extract chain --------------------------
			amps = amps[:,chain].reshape((ng,ns))
			locs = locs[:,chain].reshape((ng,ns,nd))
			stds = stds[:,chain].reshape((ng,ns,nd))
			cors = cors[:,chain].reshape((ng,ns,nd,nd))
			#--------------------------------------------
		#-------------------------------------------

		#------- Take sample ---------------
		if n_samples is not None:
			idx = np.random.choice(
				  np.arange(locs.shape[1]),
						replace=False,
						size=n_samples)

			amps = amps[:,idx]
			locs = locs[:,idx]
			stds = stds[:,idx]
			cors = cors[:,idx]
		#------------------------------------

		#------- Construct covariances ---------------
		covs = np.zeros_like(cors)
		for i,(std,cor) in enumerate(zip(stds,cors)):
			for j,(st,co) in enumerate(zip(std,cor)):
				covs[i,j] = np.diag(st).dot(
							co.dot(np.diag(st)))
		#----------------------------------------------
		#===========================================================================

		return srcs,amps,locs,covs

	def _classify(self,srcs,amps,locs,covs,names_groups):
		'''
		Obtain the class of each source at each chain step
		'''
		print("Classifying sources ...")

		if "GMM" in self.prior:
			#------- Swap axes -----------------
			pos_amps = np.swapaxes(amps,0,1)
			pos_locs = np.swapaxes(locs,0,1)
			pos_covs = np.swapaxes(covs,0,1)
			#-----------------------------------

			#------ Loop over sources ----------------------------------
			log_lk = np.zeros((srcs.shape[0],pos_amps.shape[0],pos_amps.shape[1]))
			if self.D == 1:
				for i,src in enumerate(srcs):
					for j,(dt,amps,locs,covs) in enumerate(zip(src,pos_amps,pos_locs,pos_covs)):
						for k,(amp,loc,scl) in enumerate(zip(amps,locs,np.sqrt(covs))):
							log_lk[i,j,k] = st.norm.logpdf(dt,loc=loc,scale=scl)
			else:
				for i,src in enumerate(srcs):
					for j,(dt,amps,locs,covs) in enumerate(zip(src,pos_amps,pos_locs,pos_covs)):
						for k,(amp,loc,cov) in enumerate(zip(amps,locs,covs)):
							log_lk[i,j,k] = st.multivariate_normal(mean=loc,cov=cov,
												allow_singular=True).logpdf(dt)

			idx = st.mode(log_lk.argmax(axis=2),axis=1,keepdims=True)[0].flatten()

		else:
			idx = np.zeros(len(self.ID),dtype=np.int32)

		grps = [names_groups[i] for i in idx]

		self.df_groups = pn.DataFrame(data={"group":idx,"label":grps},index=self.ID)

	def _kinematic_indices(self,group="posterior",chain=None,n_samples=None):
		'''
		Compute the kinematic indicators of expansion and rotation
		'''
		#---- Get parameters -------
		srcs,_,locs,_ = self._extract(group=group,
										n_samples=n_samples,
										chain=chain)

		if "GMM" in self.prior:
			sys.exit("Kinematic indices are not available for mixture models!")

		#=============== Sources =====================================
		#-- Extract positions and velocities-----
		srcs_pos = srcs[:,:,:3]
		srcs_vel = srcs[:,:,3:]

		locs_pos = locs[:,:,:3]
		locs_vel = locs[:,:,3:]
		#---------------------------------------

		#-- Relative positions and velocities ----
		rs = np.subtract(srcs_pos,locs_pos)
		vs = np.subtract(srcs_vel,locs_vel)
		#-----------------------------------------

		#---- Normalized position ----------------------
		nrs = np.linalg.norm(rs,axis=2,keepdims=True)
		ers = rs/nrs
		#-----------------------------------------------

		#------- Products --------------------------------
		srcs_exp = np.empty((ers.shape[0],ers.shape[1]))
		srcs_rot = np.empty((ers.shape[0],ers.shape[1]))
		for i,(er, v) in enumerate(zip(ers,vs)):
			srcs_exp[i] = np.diag(np.inner(er,v))
			srcs_rot[i] = np.linalg.norm(np.cross(er,v),axis=1)
		#------------------------------------------------------

		mean_speed = np.mean(srcs_exp,axis=1)
		norm_radii = np.mean(nrs.squeeze(),axis=1)
		#===========================================================

		#============== Clusters ===================================
		if self.velocity_model in ["constant","linear"]:
			if group == "posterior":
				data = self.ds_posterior.data_vars
			elif group == "prior":
				data = self.ds_prior.data_vars
			else:
				sys.exit("Group not recognized")

			#------------ Extract values -------------
			kappa = np.array(data["6D::kappa"].values)

			if self.velocity_model == "linear":
				omega = np.array(data["6D::omega"].values)
			#-----------------------------------------

			#----------------- Merge --------------------
			if chain is None:
				nc,ns,nd = kappa.shape
				kappa = kappa.reshape((nc*ns,nd))

				if self.velocity_model == "linear":
					nc,ns,nv,nd = omega.shape
					omega = omega.reshape((nc*ns,nv,nd))
			else:
				nc,ns,nd = kappa.shape
				kappa = kappa[chain].reshape((ns,nd))

				if self.velocity_model == "linear":
					nc,ns,nv,nd = omega.shape
					omega = omega[chain].reshape((ns,nv,nd))
			#--------------------------------------------

			#----------- Tensor----------------
			T = np.zeros((kappa.shape[0],3,3))
			T[:,0,0] = kappa[:,0]*1000.
			T[:,1,1] = kappa[:,1]*1000.
			T[:,2,2] = kappa[:,2]*1000.
			
			if self.velocity_model == "linear":
				T[:,0,1] = omega[:,0,0]*1000.
				T[:,0,2] = omega[:,0,1]*1000.
				T[:,1,2] = omega[:,0,2]*1000.
				T[:,1,0] = omega[:,1,0]*1000.
				T[:,2,0] = omega[:,1,1]*1000.
				T[:,2,1] = omega[:,1,2]*1000.
			#----------------------------------

			#--------- Indicators ------------
			exp = kappa.mean(axis=1)*1000.
			if self.velocity_model == "linear":
				omega = np.column_stack([
						0.5*(T[:,2,1]-T[:,1,2]),
						0.5*(T[:,0,2]-T[:,2,0]),
						0.5*(T[:,1,0]-T[:,0,1])])
				rot = np.linalg.norm(omega,axis=1)
			else:
				rot = np.zeros_like(exp)
			#----------------------------------

		else:
			print(
		"WARNING: the expansion and rotation indicators are computed from the dot and cross-product \
		of the positions and velocities. Instead use the linear velocity model")
			exp = srcs_exp*1000.
			rot = srcs_rot*1000.
			T = None

		return norm_radii,mean_speed,exp,rot,T

	def plot_model(self,
		file_plots=None,
		figsize=None,
		n_samples=100,
		chain=None,
		fontsize_title=16,
		labels=["X [pc]","Y [pc]","Z [pc]",
				"U [km/s]","V [km/s]","W [km/s]"],
		posterior_kwargs={"label":"Model",
						"color":"orange",
						"linewidth":1,
						"alpha":0.1},

		prior_kwargs={"label":"Prior",
						"color":"green",
						"linewidth":0.5,
						"alpha":0.1},

		source_kwargs={"label":"Source",
						"marker":"o",
						"color":"black",
						"size":2,
						"error_color":"grey",
						"error_lw":0.5,
						"cmap_mix":"tab10_r",
						"cmap_pos":"coolwarm",
						"cmap_vel":"summer"},
		groups_kwargs={"color":{"A":"tab:blue",
								"B":"tab:orange",
								"C":"tab:green",
								"D":"tab:brown",
								"Field":"tab:gray"}},
		ticks={"minor":16,"major":8},
		legend_bbox_to_anchor=(0.25, 0., 0.5, 0.5)
		):
		"""
		This function plots the model.
		"""
		if self.D == 1:
			print("Plot model valid only for 3D and 6D")
			return

		msg_n = "The required n_samples {0} is larger than those in the posterior.".format(n_samples)

		assert n_samples <= self.ds_posterior.sizes["draw"], msg_n

		print("Plotting model ...")

		file_plots = self.dir_out+"/Model.pdf" if (file_plots is None) else file_plots

		#------------ Chain ----------------------
		if "GMM" in self.prior:
			chain = 0 if chain is None else chain
			names_groups = self.ds_posterior.coords["component"].values
		else:
			names_groups = ["A"]
		#-----------------------------------------

		pdf = PdfPages(filename=file_plots)

		#---------- Extract prior and posterior --------------------------------------------
		pos_srcs,pos_amps,pos_locs,pos_covs = self._extract(group="posterior",
													n_samples=n_samples,
													chain=chain)
		if self.ds_prior is not None:
			_,_,pri_locs,pri_covs = self._extract(group="prior",n_samples=n_samples)
		#-----------------------------------------------------------------------------------

		#---------- Classify sources -------------------
		if not hasattr(self,"df_groups"):
			self._classify(pos_srcs,pos_amps,pos_locs,pos_covs,names_groups)
		#------------------------------------------------

		#-- Sources mean and standard deviation ---------
		srcs_loc = np.mean(pos_srcs,axis=1)
		srcs_std = np.std(pos_srcs,axis=1)
		#------------------------------------------------

		#======================== Colors ================================
		if "GMM" in self.prior or self.D == 3:
			#-------- Groups ---------------------------
			groups = self.df_groups["group"].to_numpy()
			#-------------------------------------------

			#------------ Colormaps ------------------------
			cmap_pos = cm.get_cmap(source_kwargs["cmap_mix"])
			cmap_vel = cm.get_cmap(source_kwargs["cmap_mix"])
			#------------------------------------------------

			#-------- Normalizations --------------------------------
			norm_pos = lambda x:x
			norm_vel = lambda x:x
			#--------------------------------------------------------

			#------- Colors of sources --------------
			srcs_clr_pos = cmap_pos(norm_pos(groups))
			srcs_clr_vel = cmap_vel(norm_vel(groups))
			#----------------------------------------

		else:
			#--------- Kinematic indices ------------------------------
			nrs,nvr,exp,rot,tensor = self._kinematic_indices(group="posterior")
			print("Expansion: {0:2.2f} +/- {1:2.2f} m.s-1.pc-1".format(
											np.mean(exp),np.std(exp)))
			print("Rotation:  {0:2.2f} +/- {1:2.2f} m.s-1.pc-1".format(
											np.mean(rot),np.std(rot)))
			#----------------------------------------------------------

			#------------ Colormaps --------------------------
			cmap_pos = cm.get_cmap(source_kwargs["cmap_pos"])
			cmap_vel = cm.get_cmap(source_kwargs["cmap_vel"])
			#-------------------------------------------------

			#------------ Normalizations ------------------------
			if nvr.min() < 0 and nvr.max() > 0:
				vcenter = 0.0
			else:
				vcenter = np.sign(nvr.min())*0.5*np.abs((nvr.max()-nvr.min()))

			norm_pos = TwoSlopeNorm(vcenter=vcenter,
								vmin=nvr.min(),vmax=nvr.max())
			norm_vel = Normalize(vmin=nrs.min(),vmax=nrs.max())
			#----------------------------------------------------

			#--------- Sources colors ------------
			srcs_clr_pos = cmap_pos(norm_pos(nvr))
			srcs_clr_vel = cmap_vel(norm_vel(nrs))
			#-------------------------------------
		#================================================================

		#=================== Positions ================================================
		fig, axs = plt.subplots(nrows=2,ncols=2,figsize=figsize)
		for ax,idx in zip([axs[0,0],axs[0,1],axs[1,0]],[[0,1],[2,1],[0,2]]):
			#--------- Sources --------------------------
			ax.errorbar(x=srcs_loc[:,idx[0]],
						y=srcs_loc[:,idx[1]],
						xerr=srcs_std[:,idx[0]],
						yerr=srcs_std[:,idx[1]],
						fmt='none',
						ecolor=source_kwargs["error_color"],
						elinewidth=source_kwargs["error_lw"],
						zorder=2)
			ax.scatter(x=srcs_loc[:,idx[0]],
						y=srcs_loc[:,idx[1]],
						c=srcs_clr_pos,
						marker=source_kwargs["marker"],
						s=source_kwargs["size"],
						zorder=2)

			#-------- Posterior ----------------------------------------------------------
			for mus,covs in zip(pos_locs,pos_covs):
				for mu,cov in zip(mus,covs):
						width, height, angle = get_principal(cov,idx)
						ell  = Ellipse(mu[idx],width=width,height=height,angle=angle,
										clip_box=ax.bbox,
										edgecolor=posterior_kwargs["color"],
										facecolor=None,
										fill=False,
										linewidth=posterior_kwargs["linewidth"],
										alpha=posterior_kwargs["alpha"],
										zorder=1)
						ax.add_artist(ell)
			#-----------------------------------------------------------------------------

			#-------- Prior ----------------------------------------------------------
			if self.ds_prior is not None:
				for mus,covs in zip(pri_locs,pri_covs):
					for mu,cov in zip(mus,covs):
							width, height, angle = get_principal(cov,idx)
							ell  = Ellipse(mu[idx],width=width,height=height,angle=angle,
											clip_box=ax.bbox,
											edgecolor=prior_kwargs["color"],
											facecolor=None,
											fill=False,
											linewidth=prior_kwargs["linewidth"],
											alpha=prior_kwargs["alpha"],
											zorder=0)
							ax.add_artist(ell)
			#-----------------------------------------------------------------------------

			#------------- Titles -------------
			ax.set_xlabel(labels[idx[0]])
			ax.set_ylabel(labels[idx[1]])
			#-----------------------------------

			#----------------- Ticks ----------------------------------------
			ax.xaxis.set_major_locator(ticker.MaxNLocator(ticks["major"]))
			# ax.xaxis.set_minor_locator(ticker.MaxNLocator(ticks["minor"]))
			ax.yaxis.set_major_locator(ticker.MaxNLocator(ticks["major"]))
			# ax.yaxis.set_minor_locator(ticker.MaxNLocator(ticks["minor"]))
			#----------------------------------------------------------------

		axs[0,0].axes.xaxis.set_visible(False)
		axs[0,1].axes.yaxis.set_visible(False)

		#------------- Legend lines  ---------------------------------------
		prior_line = mlines.Line2D([], [], color=prior_kwargs["color"], 
								marker=None, label=prior_kwargs["label"])
		group_line = mlines.Line2D([], [], color=posterior_kwargs["color"], 
								marker=None, label=posterior_kwargs["label"])
		#-------------------------------------------------------------------

		#----------- Legend symbols ----------------------------------
		if "GMM" in self.prior:
			source_mrkr =  [mlines.Line2D([], [], 
								marker=source_kwargs["marker"], color="w", 
								markerfacecolor=cmap_pos(norm_pos(row["group"])), 
								markersize=5,
								label=row["label"]) 
								for i,row in self.df_groups.drop_duplicates().iterrows()] 
		else:
			source_mrkr =  [mlines.Line2D([], [], marker=source_kwargs["marker"], color="w", 
						  markerfacecolor=source_kwargs["color"], 
						  markersize=5,
						  label=source_kwargs["label"])]
		#---------------------------------------------------------------

		if self.ds_prior is not None:
			handles = sum([[prior_line],[group_line],source_mrkr],[])
		else:
			handles = sum([[group_line],source_mrkr],[])
		axs[1,1].legend(handles=handles,loc='center',
							bbox_to_anchor=legend_bbox_to_anchor)
		axs[1,1].axis("off")
		#-------------------------------------------------------------------------------

		#--------- Colour bar---------------------------------------------------------------------
		if "GMM" not in self.prior and self.D == 6:
			fig.colorbar(cm.ScalarMappable(norm=norm_pos, cmap=cmap_pos),
								ax=axs[1,1],fraction=0.3,
								anchor=(0.0,0.0),
								shrink=0.75,extend="both",label='$||V_r||$ [km/s]')
		#-----------------------------------------------------------------------------------------

		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#==============================================================================================

		#========================= Velocities =========================================================
		if self.D == 6:
			fig, axs = plt.subplots(nrows=2,ncols=2,figsize=figsize)
			for ax,idx in zip([axs[0,0],axs[0,1],axs[1,0]],[[3,4],[5,4],[3,5]]):
				#--------- Sources --------------------------
				ax.errorbar(x=srcs_loc[:,idx[0]],
							y=srcs_loc[:,idx[1]],
							xerr=srcs_std[:,idx[0]],
							yerr=srcs_std[:,idx[1]],
							fmt='none',
							ecolor=source_kwargs["error_color"],
							elinewidth=source_kwargs["error_lw"],
							zorder=2)
				clr_vel = ax.scatter(x=srcs_loc[:,idx[0]],
							y=srcs_loc[:,idx[1]],
							c=srcs_clr_vel,
							marker=source_kwargs["marker"],
							s=source_kwargs["size"],
							zorder=2)

				#-------- Posterior ----------------------------------------------------------
				for mus,covs in zip(pos_locs,pos_covs):
					for mu,cov in zip(mus,covs):
							width, height, angle = get_principal(cov,idx)
							ell  = Ellipse(mu[idx],width=width,height=height,angle=angle,
											clip_box=ax.bbox,
											edgecolor=posterior_kwargs["color"],
											facecolor=None,
											fill=False,
											linewidth=posterior_kwargs["linewidth"],
											alpha=posterior_kwargs["alpha"],
											zorder=1)
							ax.add_artist(ell)
				#-----------------------------------------------------------------------------

				#-------- Prior ----------------------------------------------------------
				if self.ds_prior is not None:
					for mus,covs in zip(pri_locs,pri_covs):
						for mu,cov in zip(mus,covs):
								width, height, angle = get_principal(cov,idx)
								ell  = Ellipse(mu[idx],width=width,height=height,angle=angle,
												clip_box=ax.bbox,
												edgecolor=prior_kwargs["color"],
												facecolor=None,
												fill=False,
												linewidth=prior_kwargs["linewidth"],
												alpha=prior_kwargs["alpha"],
												zorder=0)
								ax.add_artist(ell)
				#-----------------------------------------------------------------------------

				#--------- Titles ------------
				ax.set_xlabel(labels[idx[0]])
				ax.set_ylabel(labels[idx[1]])
				#----------------------------

				#----------------- Ticks ----------------------------------------
				ax.xaxis.set_major_locator(ticker.MaxNLocator(ticks["major"]))
				# ax.xaxis.set_minor_locator(ticker.MaxNLocator(ticks["minor"]))
				ax.yaxis.set_major_locator(ticker.MaxNLocator(ticks["major"]))
				# ax.yaxis.set_minor_locator(ticker.MaxNLocator(ticks["minor"]))
				#----------------------------------------------------------------

			axs[0,0].axes.xaxis.set_visible(False)
			axs[0,1].axes.yaxis.set_visible(False)

			#------------- Legend lines --------------------------------------
			prior_line = mlines.Line2D([], [], color=prior_kwargs["color"], 
									marker=None, label=prior_kwargs["label"])
			group_line = mlines.Line2D([], [], color=posterior_kwargs["color"], 
									marker=None, label=posterior_kwargs["label"])
			#-----------------------------------------------------------------

			#----------- Legend symbols ----------------------------------
			if "GMM" in self.prior:
				source_mrkr =  [mlines.Line2D([], [], 
								marker=source_kwargs["marker"],
								color="w", 
								markerfacecolor=cmap_vel(norm_vel(row["group"])), 
								markersize=5,
								label=row["label"]) 
								for i,row in self.df_groups.drop_duplicates().iterrows()]
			else:
				source_mrkr = [mlines.Line2D([], [], marker=source_kwargs["marker"], 
								color="w", 
								markerfacecolor=source_kwargs["color"], 
								markersize=5,
								label=source_kwargs["label"])]
			#---------------------------------------------------------------

			#----------- Handles -------------------------------------------
			if self.ds_prior is not None:
				handles = sum([[prior_line],[group_line],source_mrkr],[])
			else:
				handles = sum([[group_line],source_mrkr],[])
			axs[1,1].legend(handles=handles,loc='center',
							bbox_to_anchor=legend_bbox_to_anchor)
			axs[1,1].axis("off")
			#--------------------------------------------------------------

			#--------- Colour bar-------------------------------------------
			if "GMM" not in self.prior:
				fig.colorbar(cm.ScalarMappable(norm=norm_vel, cmap=cmap_vel),
						ax=axs[1,1],fraction=0.3,
						anchor=(0.0,0.0),
						shrink=0.75,extend="max",label='$||r||$ [pc]')
			#--------------------------------------------------------------

			plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
									wspace=0.0, hspace=0.0)
			pdf.savefig(bbox_inches='tight')
			plt.close()
		#=============================================================================================

		pdf.close()

	def _get_map(self,var_names):
		#------------------------------------
		labeller = az.labels.BaseLabeller()
		metric_names = ["MAP"]
		#------------------------------------

		idx_map = np.unravel_index(np.argmax(
			self.trace.sample_stats.lp.values),
			shape=self.trace.sample_stats.lp.values.shape)
		
		data = az.extract(self.ds_posterior,var_names=var_names)
		data_map = az.utils.get_coords(data,
			{"chain":idx_map[0],"draw":idx_map[1]})
		
		joined = data_map.assign_coords(metric=metric_names).reset_coords(drop=True)
		n_metrics = len(metric_names)
		n_vars = np.sum([joined[var].size // n_metrics for var in joined.data_vars])

		summary_df = pn.DataFrame(
			(np.full((cast(int, n_vars), n_metrics), np.nan)), columns=metric_names
		)
		indices = []
		for i, (var_name, sel, isel, values) in enumerate(
			az.sel_utils.xarray_var_iter(joined, skip_dims={"metric"})
		):
			summary_df.iloc[i] = values
			indices.append(labeller.make_label_flat(var_name, sel, isel))
		summary_df.index = indices

		return summary_df

	def save_statistics(self,hdi_prob=0.95,chain_gmm=[0],stat_focus="mean"):
		'''
		Saves the statistics to a csv file.
		Arguments:
		
		'''
		print("Computing statistics ...")

		#----------------------- Functions ---------------------------------
		def distance(x,y,z):
			return np.sqrt(x**2 + y**2 + z**2)
		#---------------------------------------------------------------------
		
		#--------- Coordinates -------------------------
		# In MM use only one chain
		if "GMM" in self.prior:
			print("WARNING: In mixture models only one "\
				+"chain is used to compute statistics.\n"\
				+"Set chain_gmm=[0,1,..,n_chains] to override.")
			data = az.utils.get_coords(self.ds_posterior,{"chain":chain_gmm})
			names_groups = self.ds_posterior.coords["component"].values
		else:
			data = self.ds_posterior
			names_groups = ["A"]
		#------------------------------------------------------------
		
		#--------- Get MAP ------------------------------------------
		df_map_grp = self._get_map(var_names=self.stats_variables)
		df_map_src = self._get_map(var_names=[self.source_variables])
		#-------------------------------------------------------------

		#-------------- Source statistics ----------------------------
		source_csv = self.dir_out +"/Sources_statistics.csv"
		df_source  = az.summary(data,var_names=self.source_variables,
						stat_focus = stat_focus,
						hdi_prob=hdi_prob,
						extend=True)
		df_source = df_map_src.join(df_source)
		#--------------------------------------------------------------

		#------------- Replace parameter id by source ID----------------
		n_sources = len(self.ID)
		ID  = np.repeat(self.ID,self.D,axis=0)
		idx = np.tile(np.arange(self.D),n_sources)

		df_source.set_index(ID,inplace=True)
		df_source.insert(loc=0,column="parameter",value=idx)
		#---------------------------------------------------------------

		if self.D in [1,3,6] :
			#---------- Classify sources -------------------
			if not hasattr(self,"df_groups"):
				if self.ds_posterior.sizes["draw"] > 100:
					n_samples = 100
				else:
					n_samples = self.ds_posterior.sizes["draw"]

				#------- Extract GMM parameters ----------------------------------
				pos_srcs,pos_amps,pos_locs,pos_covs = self._extract(group="posterior",
											n_samples=n_samples,
											chain=chain_gmm)
				#-----------------------------------------------------------------

				self._classify(pos_srcs,pos_amps,pos_locs,pos_covs,names_groups)
			#------------------------------------------------

		
			# ------ Parameters into columns ------------------------
			dfs = []
			for i in range(self.D):
				idx = np.where(df_source["parameter"] == i)[0]
				tmp = df_source.drop(columns="parameter").add_suffix(
								"_"+self.names_coords[i])
				dfs.append(tmp.iloc[idx])

			#-------- Join on index --------------------
			df_source = dfs[0]
			for i in range(1,self.D) :
				df_source = df_source.join(dfs[i],
					how="inner",lsuffix="",rsuffix="_"+self.names_coords[i])
			#---------------------------------------------------------------------

			#---------- Add group -----------------------------------
			df_source = df_source.join(self.df_groups)
			#----------------------------------------------

			if self.D > 1:
				#------ Add distance ---------------------------------------------------------
				df_source["MAP_distance"] = df_source[["MAP_X","MAP_Y","MAP_Z"]].apply(
					lambda x: distance(*x),axis=1)

				df_source["mean_distance"] = df_source[["mean_X","mean_Y","mean_Z"]].apply(
					lambda x: distance(*x),axis=1)
				#----------------------------------------------------------------------------

		#---------- Save source data frame ----------------------
		df_source.to_csv(path_or_buf=source_csv,index_label=self.id_name)

		#-------------- Global statistics ----------------------------------
		if len(self.cluster_variables) > 0:
			grp_csv = self.dir_out +"/Cluster_statistics.csv"
			df_grp = az.summary(data,var_names=self.stats_variables,
							stat_focus=stat_focus,
							hdi_prob=hdi_prob,
							round_to=3,
							extend=True)
			df_grp = df_map_grp.join(df_grp)

			df_grp.to_csv(path_or_buf=grp_csv,index_label="Parameter")
		#-------------------------------------------------------------------

		#--------------- Velocity field ----------------------------------
		if "6D::kappa" in self.cluster_variables:
			field_csv = self.dir_out +"/Linear_velocity_statistics.csv"
			_,_,exp,rot,T = self._kinematic_indices(group="posterior")

			df_field = az.summary(data={
				"Exp [m.s-1.pc-1]":exp,
				"Rot [m.s-1.pc-1]":rot,
				"Txx [m.s-1.pc-1]":T[:,0,0],
				"Txy [m.s-1.pc-1]":T[:,0,1],
				"Txz [m.s-1.pc-1]":T[:,0,2],
				"Tyx [m.s-1.pc-1]":T[:,1,0],
				"Tyy [m.s-1.pc-1]":T[:,1,1],
				"Tyz [m.s-1.pc-1]":T[:,1,2],
				"Tzx [m.s-1.pc-1]":T[:,2,0],
				"Tzy [m.s-1.pc-1]":T[:,2,1],
				"Tzz [m.s-1.pc-1]":T[:,2,2]
				},
							stat_focus=stat_focus,
							hdi_prob=hdi_prob,
							kind="stats",
							extend=True)

			df_field.to_csv(path_or_buf=field_csv,index_label="Parameter")

			#----------- Notation as in Lindegren et al. 2000 --------------
			lenn_csv = self.dir_out +"/Lindegren_velocity_statistics.csv"
			df_Lenn = az.summary(data={
				"kappa [m.s-1.pc-1]":exp,
				"kappa_x [m.s-1.pc-1]":T[:,0,0],
				"kappa_y [m.s-1.pc-1]":T[:,1,1],
				"kappa_z [m.s-1.pc-1]":T[:,2,2],
				"omega_x [m.s-1.pc-1]":0.5*(T[:,2,1]-T[:,1,2]),
				"omega_y [m.s-1.pc-1]":0.5*(T[:,0,2]-T[:,2,0]),
				"omega_z [m.s-1.pc-1]":0.5*(T[:,1,0]-T[:,0,1]),
				"w_1 [m.s-1.pc-1]":0.5*(T[:,2,1]+T[:,1,2]),
				"w_2 [m.s-1.pc-1]":0.5*(T[:,0,2]+T[:,2,0]),
				"w_3 [m.s-1.pc-1]":0.5*(T[:,1,0]+T[:,0,1]),
				"w_4 [m.s-1.pc-1]":T[:,0,0],
				"w_5 [m.s-1.pc-1]":T[:,1,1]
				},
							stat_focus=stat_focus,
							hdi_prob=hdi_prob,
							kind="stats",
							extend=True)

			df_Lenn.to_csv(path_or_buf=lenn_csv,index_label="Parameter")

		#-------------------------------------------------------------------

	def save_samples(self,merge=True):
		'''
		Saves the chain samples to an h5 file.
		Arguments:
		dir_csv (string) Directory where to save the samples
		merge:: True # Merge chains into single dimension
		'''
		print("Saving samples ...")

		#------- Get IDs -----------------------
		IDs = pn.read_csv(self.file_ids)[self.id_name].values.astype('str')
		#---------------------------------------

		#------ Open h5 file -------------------
		file_h5 = self.dir_out + "/Samples.h5"

		sources_trace = self.ds_posterior[self.source_variables].to_array()

		with h5py.File(file_h5,'w') as hf:
			grp_glb = hf.create_group("Cluster")
			grp_src = hf.create_group("Sources")

			#------ Loop over global parameters ---
			for name in self.cluster_variables:
				data = np.array(self.ds_posterior[name])
				if merge:
					data = data.reshape((data.shape[0]*data.shape[1],-1))
				grp_glb.create_dataset(name, data=data)

			#------ Loop over source parameters ---
			for i,name in enumerate(IDs):
				data = sources_trace.sel(source_id=name).to_numpy()
				if merge:
					data = data.reshape((-1,self.D))
				grp_src.create_dataset(name, data=data)

	def save_posterior_predictive(self,
		file_chains=None):
		var_name = str(self.D)+"D::true"

		file_chains = self.file_chains if (file_chains is None) else file_chains
		file_base = self.dir_out+"/posterior_predictive"

		#--------------- Extract observables -----------------------------------------------
		dfg = self.trace.posterior[var_name].to_dataframe().groupby("observable")
		dfs = []
		for obs,df in dfg.__iter__():
			df.reset_index("observable",drop=True,inplace=True)
			df.rename(columns={var_name:obs},inplace=True)
			dfs.append(df)
		df = pn.concat(dfs,axis=1,ignore_index=False)
		#-----------------------------------------------------------------------------------

		#--------- Save H5 samples ---------------------------
		df.to_hdf(file_base + ".h5",key="posterior_predictive")
		#-----------------------------------------------------

		#---------- Groupby source id ------------------------
		dfg = df.groupby("source_id")

		dfs = []
		for name, df in dfg.__iter__():
			tmp = pn.merge(
						left=df.mean(axis=0).to_frame().T,
						right=df.std(axis=0).to_frame().T,
						left_index=True,right_index=True,
						suffixes=("","_error")).set_index(
						np.array(name).reshape(1)).rename_axis(
						index="source_id")
			dfs.append(tmp)
		df = pn.concat(dfs,axis=0,ignore_index=False)
		#-------------------------------------------------------
		
		#------------ Save to CSV ---------------
		df.to_csv(file_base + ".csv",index=True)
		#-----------------------------------------
		

	def evidence(self,N_samples=None,M_samples=1000,dlogz=1.0,nlive=None,
		quantiles=[0.05,0.95],
		print_progress=False,
		plot=False):

		assert self.D == 1, "Evidence is only implemented for dimension 1."

		#------ Add media to quantiles ---------------
		quantiles = [quantiles[0],0.5,quantiles[1]]
		print(50*"=")
		print("Estimating evidence of prior: ",self.prior)

		#------- Initialize evidence module ----------------
		dyn = Evidence1D(self.mu_data,self.sg_data,
				prior=self.prior,
				parameters=self.parameters,
				hyper_alpha=self.hyper["alpha"],
				hyper_beta=self.hyper["beta"],
				hyper_gamma=self.hyper["gamma"],
				hyper_delta=self.hyper["delta"],
				N_samples=N_samples,
				M_samples=M_samples,
				transformation=self.transformation,
				quantiles=quantiles)
		#  Compute evidence 
		results = dyn.run(dlogz=dlogz,nlive=nlive,print_progress=print_progress)

		logZ    = results["logz"][-1]
		logZerr = results["logzerr"][-1]

		print("Log Z: {0:.3f} +/- {1:.3f}".format(logZ,logZerr))
		print(50*"=")

		evidence   = pn.DataFrame(data={"lower":logZ-logZerr,"median":logZ,"upper":logZ+logZerr}, index=["logZ"])
		parameters = dyn.parameters_statistics(results)
		summary    = parameters.append(evidence)

		file = self.dir_out +"/Evidence.csv"

		summary.to_csv(file,index_label="Parameter")

		if plot:
			dyn.plots(results,file=file.replace(".csv",".pdf"))
		
		return

		


