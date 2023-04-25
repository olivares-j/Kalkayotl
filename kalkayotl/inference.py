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
import sys
import random
import pymc as pm
import numpy as np
import pandas as pn
import xarray
import arviz as az
import h5py
import scipy.stats as st
from scipy.linalg import inv as inverse
from string import ascii_uppercase
from astropy.stats import circmean
from astropy import units as u
import pymc.sampling_jax
import pytensor.tensor as at
from typing import cast

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
from kalkayotl.Transformations import Iden,pc2mas # 1D
from kalkayotl.Transformations import icrs_xyz_to_radecplx,galactic_xyz_to_radecplx #3D
from kalkayotl.Transformations import icrs_xyzuvw_to_astrometry_and_rv #6D
from kalkayotl.Transformations import galactic_xyzuvw_to_astrometry_and_rv #6D
#------------------------------------------------------------------------


class Inference:
	"""
	This class provides flexibility to infer the distance distribution given the parallax and its uncertainty
	"""
	def __init__(self,dimension,
				dir_out,
				zero_points,
				indep_measures=False,
				reference_system=None,
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

		self.D                = dimension 
		self.zero_points      = zero_points
		self.dir_out          = dir_out
		self.indep_measures   = indep_measures
		
		self.reference_system = reference_system
		self.file_ids         = self.dir_out+"/Identifiers.csv"
		self.file_obs         = self.dir_out+"/Observations.nc"
		self.file_chains      = self.dir_out+"/Chains.nc"

		self.asec2deg = 1.0/(60.*60.)

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


	def load_data(self,file_data,corr_func="Lindegren+2020",*args,**kwargs):
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

		#-------- Drop NaNs ------------------------
		data.dropna(subset=self.names_nan,inplace=True,
						thresh=len(self.names_nan))
		#----------------------------------------------

		#--------- mas to degrees -------------------------
		# Fixes RA Dec uncertainties to 1 arcsec
		data["ra_error"]  = 1.*self.asec2deg
		data["dec_error"] = 1.*self.asec2deg
		#--------------------------------------------------

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

		#-------- Compute inverse of covariance matrix --------------------
		self.idx_data  = idx_obs
		self.mu_data  = mu_data
		self.sg_data  = sg_data
		self.tau_data = np.linalg.inv(sg_data)
		#=================================================================================

		print("Data correctly loaded")


	def setup(self,prior,
				parameters,
				hyper_parameters,
				parametrization,
				sampling_space="physical",
				reference_system="Galactic",
				velocity_model="joint"
				):
		'''
		Set-up the model with the corresponding dimensions and data
		'''

		self.prior            = prior
		self.parameters       = parameters
		self.hyper            = hyper_parameters
		self.parametrization  = parametrization
		self.velocity_model   = velocity_model
		self.sampling_space   = sampling_space

		print(15*"+", " Prior setup ", 15*"+")
		print("Type of prior: ",self.prior)

		msg_alpha = "hyper_alpha must be specified."
		msg_beta  = "hyper_beta must be specified."
		msg_trans = "Transformation must be either pc or mas."
		msg_delta = "hyper_delta must be specified."
		msg_gamma = "hyper_gamma must be specified."
		msg_nu    = "hyper_nu must be specified."
		msg_central = "Only the central parametrization is valid for this configuration."
		msg_non_central = "Only the non-central parametrization is valid for this configuration."
		msg_weights = "weights must be greater than 5%."

		assert sampling_space in ["observed","physical"], "ERROR: sampling space can only be physical or observed"
		assert velocity_model in ["joint","constant","linear"], "ERROR: velocity_model not recognized!"

		if self.D in [3,6]:
			assert sampling_space == "physical", "3D and 6D models work only in the physical space."

		#============= Transformations ====================================
		if reference_system == "ICRS":
			if self.D == 3:
				Transformation = icrs_xyz_to_radecplx
			elif self.D == 6:
				Transformation = icrs_xyzuvw_to_astrometry_and_rv
			elif self.D == 1:
				if sampling_space == "physical":
					Transformation = pc2mas
				else:
					Transformation = Iden
			else:
				sys.exit("ERROR:Uknown")

		elif reference_system == "Galactic":
			if self.D == 3:
				Transformation = galactic_xyz_to_radecplx
			elif self.D == 6:
				Transformation = galactic_xyzuvw_to_astrometry_and_rv
			elif self.D == 1:
				if sampling_space == "physical":
					Transformation = pc2mas
				else:
					Transformation = Iden
		else:
			sys.exit("Reference system not accepted")
		#==================================================================

		#============== Mixtures =====================================================
		if "GMM" in self.prior:

			if self.prior == "FGMM":
				n_components = 2
			else:
				n_components = self.hyper["n_components"]

			if self.parameters["weights"] is None:
				assert self.hyper["delta"] is not None, msg_delta
			else:
				#-------------- Read from input file ----------------------
				if isinstance(self.parameters["weights"],str):
					#---- Extract scale parameters ------------
					wgh = pn.read_csv(self.parameters["weights"],
								usecols=["Parameter","mode"])
					wgh = wgh[wgh["Parameter"].str.contains("weights")]
					#------------------------------------------

					#---- Set weights ----------------------------
					self.parameters["weights"] = wgh["mode"].values
					#-----------------------------------------------
				#-----------------------------------------------------------

				#--------- Verify weights ---------------------------------
				print("The weights parameter is fixed to:")
				print(self.parameters["weights"])
				assert len(self.parameters["weights"]) == n_components, \
					"The size of the weights parameter is incorrect!"
				assert np.min(self.parameters["weights"])> 0.05, msg_weights
				#-----------------------------------------------------------

			assert self.parametrization == "central", msg_central

			if self.prior == "FGMM":
				msg_fgmm = "In FGMM there are only two components"
				assert len(self.hyper["delta"]) == 2, msg_fgmm
		#============================================================================

		#====================== Location ==========================================================
		if self.parameters["location"] is None:
			#---------------- Alpha ---------------------------------------------------------
			if self.hyper["alpha"] is None:
				print("The alpha hyper-parameter has been set to:")

				#-- Cluster dispersion ----
				uvw_sd = 5.
				xyz_fc = 0.2
				#-------------------------

				#------------------------ Cluster mean value ------------------------
				if self.D == 1:
					#---------- Mean distance ------------
					d = 1000./self.mean_observed[0]
					#------------------------------------

					#---------- Dispersion -----------------
					d_sd = xyz_fc*d
					#---------------------------------------

					self.hyper["alpha"] = {"loc":d,"scl":d_sd}
					print("D: {0:2.1f} +/- {1:2.1f}".format(d,d_sd))

				
				elif self.D == 3:
					#------------ Cluster mean coordinates -------------------
					x,y,z,_,_,_ = astrometry_and_rv_to_phase_space(np.append(
									self.mean_observed,[0.0,0.0,0.0])[np.newaxis,:],
									reference_system=self.reference_system).flatten()
					#----------------------------------------------------------

					#---------- Dispersion -----------------
					xyz_sd = xyz_fc*np.sqrt(x**2 + y**2 + z**2)
					#---------------------------------------

					self.hyper["alpha"] = {
						"loc":[x,y,z],
						"scl":[xyz_sd,xyz_sd,xyz_sd]}

					names = ["X","Y","Z"]
					locs = self.hyper["alpha"]["loc"]
					scls = self.hyper["alpha"]["scl"]
					for i,(loc,scl) in enumerate(zip(locs,scls)):
						print("{0}: {1:2.1f} +/- {2:2.1f}".format(names[i],loc,scl))

				elif self.D == 6:
					#------------ Cluster mean coordinates -------------------
					x,y,z,u,v,w = astrometry_and_rv_to_phase_space(
									self.mean_observed[np.newaxis,:],
									reference_system=self.reference_system).flatten()
					#----------------------------------------------------------

					#---------- Dispersion -----------------
					xyz_sd = xyz_fc*np.sqrt(x**2 + y**2 + z**2)
					#---------------------------------------

					self.hyper["alpha"] = {
					"loc":[x,y,z,u,v,w],
					"scl":[xyz_sd,xyz_sd,xyz_sd,uvw_sd,uvw_sd,uvw_sd]}

					names = ["X","Y","Z","U","V","W"]
					locs = self.hyper["alpha"]["loc"]
					scls = self.hyper["alpha"]["scl"]
					for i,(loc,scl) in enumerate(zip(locs,scls)):
						print("{0}: {1:2.1f} +/- {2:2.1f}".format(names[i],loc,scl))
				#----------------------------------------------------------------------
				
			#---------------------------------------------------------------------------------

		#-------------- Read from input file ----------------------------------
		else:
			if isinstance(self.parameters["location"],str):
				#---- Extract parameters ----------------------
				loc = pn.read_csv(self.parameters["location"],
							usecols=["Parameter","mode"])
				loc = loc[loc["Parameter"].str.contains("loc")]
				#----------------------------------------------

				#-------- Extraction is prior dependent ----------------
				if "GMM" in self.prior:
					assert int(loc.shape[0]/self.D) == n_components,\
					"Mismatch in the number of components"

					values = []
					for i in range(n_components):
						selection = loc["Parameter"].str.contains(
									"[{0}]".format(i),regex=False)
						values.append(loc.loc[selection,"mode"].values)

					self.parameters["location"] = values

				else:
					#---- Set location  ----------------------------
					self.parameters["location"] = loc["mode"].values
					#-----------------------------------------------
				#----------------------------------------------------------

			#--------- Verify location ---------------------------------
			print("The location parameter is fixed to:")
			if "GMM" in self.prior:
				for i,loc in enumerate(self.parameters["location"]):
					print(i,loc)
					assert len(loc) == self.D, \
						"The location parameter's size is incorrect!"
			else:
				print(self.parameters["location"])
				assert len(self.parameters["location"]) == self.D, \
					"The location parameter's size is incorrect!"
			#-----------------------------------------------------------
		#-----------------------------------------------------------------------
		#==============================================================================================
		
		#============================= Scale ===========================================================
		if self.parameters["scale"] is None:
			if self.hyper["beta"] is None:
				self.hyper["beta"] = np.array([10.0,10.0,10.0,2.0,2.0,2.0])[:self.D]
				print("The beta hyper-parameter has been set to:")
				print(self.hyper["beta"])
		else:
			#-------------- Read from input file -----------------------------
			if isinstance(self.parameters["scale"],str):
				#---- Extract scale parameters ------------
				scl = pn.read_csv(self.parameters["scale"],
							usecols=["Parameter","mode"])
				scl.fillna(value=1.0,inplace=True)
				#------------------------------------------

				#-------- Extraction is prior dependent ----------------
				if "GMM" in self.prior:
					stds = []
					cors = []
					covs = []
					for i in range(n_components):
						#---------- Select component parameters --------
						mask_std = scl["Parameter"].str.contains(
									"{0}_stds".format(i),regex=False)
						mask_cor = scl["Parameter"].str.contains(
									"{0}_corr".format(i),regex=False)
						#-----------------------------------------------

						#------Extract parameters -------------------
						std = scl.loc[mask_std,"mode"].values
						cor = scl.loc[mask_cor,"mode"].values
						#--------------------------------------------

						stds.append(std)

						#---- Construct covariance --------------
						std = np.diag(std)
						cor = np.reshape(cor,(self.D,self.D))
						cov = np.dot(std,cor.dot(std))
						#-----------------------------------------

						#--- Append -------
						cors.append(cor)
						covs.append(cov)
						#------------------
			
					self.parameters["stds"] = stds
					self.parameters["corr"] = cors
					self.parameters["scale"] = covs

				else:
					#---------- Select component parameters ---------
					mask_stds = scl["Parameter"].str.contains('stds')
					mask_corr = scl["Parameter"].str.contains('corr')
					#-------------------------------------------------

					#---- Extract parameters ----------------------
					stds = scl.loc[mask_stds,"mode"].values
					corr = scl.loc[mask_corr,"mode"].values
					#----------------------------------------------

					#---- Construct covariance --------------
					stds = np.diag(stds)
					corr = np.reshape(corr,(self.D,self.D))
					cov = np.dot(stds,corr.dot(stds))
					#-----------------------------------------

					self.parameters["scale"] = cov
				#-----------------------------------------------------
			#---------------------------------------------------------------------

			#--------- Verify scale ----------------------------------------
			print("The scale parameter is fixed to:")
			if "GMM" in self.prior:
				for i,scl in enumerate(self.parameters["scale"]):
					print(i,scl)
					assert scl.shape == (self.D,self.D), \
						"The scale parameter's shape is incorrect!"
			else:
				print(self.parameters["scale"])
				assert self.parameters["scale"].shape == (self.D,self.D), \
					"The scale parameter's shape is incorrect!"
			#----------------------------------------------------------------
		#==============================================================================================

		#==================== Miscelaneous ============================================================
		if self.parameters["scale"] is None and self.hyper["eta"] is None:
			self.hyper["eta"] = 1.0
			print("The eta hyper-parameter has been set to:")
			print(self.hyper["eta"])

		if self.prior in ["EDSD","Uniform","EFF","King"]:
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
			assert self.parameters["field_scale"] is not None, "Must specify the typical size of the field"
			assert len(self.parameters["field_scale"]) == self.D, "Size of field_scale must match model dimension"

		if self.prior in ["King","EFF"]:
			if self.prior == "KING" and self.parameters["rt"] is None:
				assert self.hyper["gamma"] is not None, msg_gamma
			if self.prior == "EFF" and self.parameters["gamma"] is None:
				assert self.hyper["gamma"] is not None, msg_gamma
		else:
			self.hyper["gamma"] = None
		#===========================================================================================================


		if self.D == 1:
			self.Model = Model1D(
								n_sources=self.n_sources,
								mu_data=self.mu_data,
								tau_data=self.tau_data,
								dimension=self.D,
								prior=self.prior,
								parameters=self.parameters,
								hyper_alpha=self.hyper["alpha"],
								hyper_beta=self.hyper["beta"],
								hyper_gamma=self.hyper["gamma"],
								hyper_delta=self.hyper["delta"],
								hyper_nu=self.hyper["nu"],
								transformation=Transformation,
								parametrization=self.parametrization,
								identifiers=self.ID,
								coordinates=self.names_coords,
								observables=self.names_mu)

		elif self.D in [3,6] and velocity_model == "joint":
			self.Model = Model3D6D(
								n_sources=self.n_sources,
								mu_data=self.mu_data,
								tau_data=self.tau_data,
								idx_observed=self.idx_data,
								dimension=self.D,
								prior=self.prior,
								parameters=self.parameters,
								hyper_alpha=self.hyper["alpha"],
								hyper_beta=self.hyper["beta"],
								hyper_gamma=self.hyper["gamma"],
								hyper_delta=self.hyper["delta"],
								hyper_eta=self.hyper["eta"],
								hyper_nu=self.hyper["nu"],
								transformation=Transformation,
								parametrization=self.parametrization,
								identifiers=self.ID,
								coordinates=self.names_coords,
								observables=self.names_mu)

		elif self.D == 6 and velocity_model != "joint":
			self.Model = Model6D_linear(n_sources=self.n_sources,
								mu_data=self.mu_data,
								tau_data=self.tau_data,
								idx_data=self.idx_data,
								prior=self.prior,
								parameters=self.parameters,
								hyper_alpha=self.hyper["alpha"],
								hyper_beta=self.hyper["beta"],
								hyper_gamma=self.hyper["gamma"],
								hyper_delta=self.hyper["delta"],
								hyper_eta=self.hyper["eta"],
								hyper_nu=self.hyper["nu"],
								transformation=Transformation,
								parametrization=self.parametrization,
								velocity_model=self.velocity_model,
								identifiers=self.ID,
								coordinates=self.names_coords,
								observables=self.names_mu)
		else:
			sys.exit("Non valid dimension or velocity model!")

		print((30+13)*"+")

	def run(self,sample_iters,tuning_iters,
		target_accept=0.8,
		chains=None,cores=None,
		step=None,
		file_chains=None,
		init_method="advi+adapt_diag",
		init_iters=int(5e5),
		prior_predictive=True,
		posterior_predictive=False,
		progressbar=True,
		nuts_sampler="pymc",
		init_absolute_tol=1e-1,
		init_relative_tol=1e-1,
		random_seed=None):
		"""
		Performs the MCMC run.
		Arguments:
		sample_iters (integer):    Number of MCMC iterations.
		tuning_iters (integer):    Number of burning iterations.
		"""

		file_chains = self.file_chains if (file_chains is None) else file_chains

		#================== Optimization ======================================================		
		random_seed_list = pymc.util._get_seeds_per_chain(random_seed, chains)

		cb = [pm.callbacks.CheckParametersConvergence(
				tolerance=init_absolute_tol, diff="absolute"),
			  pm.callbacks.CheckParametersConvergence(
			  	tolerance=init_relative_tol, diff="relative")]

		with self.Model:
			initvals = pm.find_MAP(include_transformed=False,
						maxeval=int(1e5),progressbar=False)

		initial_points = pymc.sampling.mcmc._init_jitter(
			self.Model,
			initvals,
			seeds=random_seed_list,
			jitter="jitter" in init_method,
			jitter_max_retries=10
			)

		approx = pm.fit(
			random_seed=random_seed_list[0],
			n=init_iters,
			method="advi",
			model=self.Model,
			callbacks=cb,
			progressbar=True,
			test_optimizer=pm.adagrad#_window
			)

		approx_sample = approx.sample(
			draws=chains, 
			random_seed=random_seed_list[0],
			return_inferencedata=False
			)

		initial_points = [approx_sample[i] for i in range(chains)]
		std_apoint = approx.std.eval()
		cov = std_apoint**2
		mean = approx.mean.get_value()
		weight = 10
		potential = pymc.step_methods.hmc.quadpotential.QuadPotentialDiagAdapt(
					len(cov), mean, cov, weight)

		#------------- Plot Loss ----------------------------------
		plt.figure()
		plt.plot(approx.hist)
		plt.xlabel("Iterations")
		plt.yscale("log")
		plt.ylabel("Average Loss")
		plt.savefig(self.dir_out+"/Initializations.png")
		plt.close()
		#-----------------------------------------------------------

		#------------------ Save initial point ------------------------------
		df = pn.DataFrame(data=initial_points[0]["{0}D::true".format(self.D)],
			columns=self.names_mu)
		df.to_csv(self.dir_out+"/initial_point.csv",index=False)
		#---------------------------------------------------------------------

		#--------------- Prepare step -------------------
		step = pm.NUTS(
				potential=potential,
				model=self.Model,
				target_accept=target_accept)
		#----------------------------------------------

		# -------- Fix problem with initial solution of cholesky cov-packed ----------
		name_ccp = "_cholesky-cov-packed__" 
		for vals in initial_points:
			for key,value in vals.copy().items():
				if name_ccp in key:
					del vals[key]
		# TO BE REMOVED once pymc5 solves this issue
		#----------------------------------------------------------------------------

		print("Sampling the model ...")

		with self.Model:

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
				return_inferencedata=True
				)
			#--------------------------------

			#-------- Posterior predictive -----------------------------
			if posterior_predictive:
				posterior_pred = pm.sample_posterior_predictive(trace,
						var_names=[self.Model.name+"::true"])
				trace.extend(posterior_pred)
			#--------------------------------------------------------

			#-------- Prior predictive ----------------------------------
			if prior_predictive:
				prior_pred = pm.sample_prior_predictive(
							samples=sample_iters)
				trace.extend(prior_pred)
			#-------------------------------------------------------------

			#--------- Save with arviz ------------
			az.to_netcdf(trace,file_chains)
			#-------------------------------------


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
			self.trace = az.from_netcdf(file_chains)
		except ValueError:
			sys.exit("ERROR at loading {0}".format(file_chains))
		#------------------------------------------------------------------------
		
		#---------Load posterior ---------------------------------------------------
		try:
			self.ds_posterior = self.trace.posterior
		except ValueError:
			sys.exit("There is no posterior group in {0}".format(file_chains))
		#------------------------------------------------------------------------

		#----------- Load prior -------------------------------------------------
		try:
			self.ds_prior = self.trace.prior
		except:
			self.ds_prior = None
		#-------------------------------------------------------------------------

		#------- Variable names -----------------------------------------------------------
		source_variables = list(filter(lambda x: "source" in x, self.ds_posterior.data_vars))
		cluster_variables = list(filter(lambda x: ( ("loc" in x) 
											or ("corr" in x)
											or ("stds" in x)
											or ("std" in x)
											or ("weights" in x)
											or ("beta" in x)
											or ("gamma" in x)
											or ("rt" in x)
											or ("kappa" in x)
											or ("omega" in x)
											or ("nu" in x)),
											self.ds_posterior.data_vars))
	
		trace_variables = cluster_variables.copy()
		stats_variables = cluster_variables.copy()
		tensor_variables= cluster_variables.copy()
		cluster_loc_var = cluster_variables.copy()
		cluster_std_var = cluster_variables.copy()
		cluster_cor_var = cluster_variables.copy()

		#----------- Case specific variables -------------
		tmp_srces = source_variables.copy()
		tmp_plots = cluster_variables.copy()
		tmp_stats = cluster_variables.copy()
		tmp_loc   = cluster_variables.copy()
		tmp_stds  = cluster_variables.copy()
		tmp_corr  = cluster_variables.copy()

		for var in tmp_srces:
			if "_pos" in var or "_vel" in var:
				source_variables.remove(var)

		for var in tmp_plots:
			if self.D in [3,6]:
				if "corr" in var:
					trace_variables.remove(var)
				if "lnv" in var and "stds" not in var:
					trace_variables.remove(var)

		for var in tmp_stats:
			if self.D in [3,6]:
				if not ("loc" in var 
					or "stds" in var
					or "weights" in var
					or "corr" in var 
					or "omega" in var
					or "kappa" in var):
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

		#----------------------------------------------------

		self.source_variables  = source_variables
		self.cluster_variables = cluster_variables
		self.trace_variables   = trace_variables
		self.stats_variables   = stats_variables
		self.loc_variables     = cluster_loc_var
		self.std_variables     = cluster_std_var
		self.cor_variables     = cluster_cor_var
		self.chk_variables     = sum([cluster_loc_var,cluster_std_var],[])

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
		file_chains=None,
		figsize=None,
		):
		"""
		This function plots the prior and posterior distributions.
		"""

		print("Plotting checks ...")
		file_chains = self.file_chains if (file_chains is None) else file_chains
		file_plots = self.dir_out+"/Prior_check.pdf" if (file_plots is None) else file_plots

		trace = az.from_netcdf(file_chains)
		pdf = PdfPages(filename=file_plots)
		for var in self.chk_variables:
			plt.figure(0,figsize=figsize)
			az.plot_dist_comparison(trace,var_names=var)
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

		#------ Organize sources ---------
		srcs = srcs.squeeze(axis=0)
		srcs = np.moveaxis(srcs,2,0)
		if self.D == 1:
			srcs = np.expand_dims(srcs,3)
		#----------------------------------

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
			stds = np.array(self.parameters["stds"])
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
		#--------------------------------------------------------------------

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
			T[:,0,0] = kappa[:,0]
			T[:,1,1] = kappa[:,1]
			T[:,2,2] = kappa[:,2]
			
			if self.velocity_model == "linear":
				T[:,0,1] = omega[:,0,0]
				T[:,0,2] = omega[:,0,1]
				T[:,1,2] = omega[:,0,2]
				T[:,1,0] = omega[:,1,0]
				T[:,2,0] = omega[:,1,1]
				T[:,2,1] = omega[:,1,2]
			#----------------------------------

			#--------- Indicators ------------
			exp = kappa.mean(axis=1)
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
			exp = srcs_exp
			rot = srcs_rot
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
		assert self.D in [3,6], "Only valid for 3D and 6D models"

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
			print("Expansion: {0:2.2f} +/- {1:2.2f} km/s".format(
											np.mean(exp),np.std(exp)))
			print("Rotation:  {0:2.2f} +/- {1:2.2f} km/(s pc)".format(
											np.mean(rot),np.std(rot)))
			#----------------------------------------------------------

			#------------ Colormaps --------------------------
			cmap_pos = cm.get_cmap(source_kwargs["cmap_pos"])
			cmap_vel = cm.get_cmap(source_kwargs["cmap_vel"])
			#-------------------------------------------------

			#------------ Normalizations ------------------------
			if nvr.min() > 0:
				vcenter = 0.5*(nvr.max()-nvr.min())
			else:
				vcenter = 0.0

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

		if self.D in [3,6] :
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
							extend=True)
			df_grp = df_map_grp.join(df_grp)

			df_grp.to_csv(path_or_buf=grp_csv,index_label="Parameter")
		#-------------------------------------------------------------------

		#--------------- Velocity field ----------------------------------
		if "6D::kappa" in self.cluster_variables:
			field_csv = self.dir_out +"/Linear_velocity_statistics.csv"
			_,_,exp,rot,T = self._kinematic_indices(group="posterior")

			df_field = az.summary(data={
				"Exp":exp,"Rot":rot,
				"Txx":T[:,0,0],"Txy":T[:,0,1],"Txz":T[:,0,2],
				"Tyx":T[:,1,0],"Tyy":T[:,1,1],"Tyz":T[:,1,2],
				"Tzx":T[:,2,0],"Tzy":T[:,2,1],"Tzz":T[:,2,2],
				},
							stat_focus=stat_focus,
							hdi_prob=hdi_prob,
							kind="stats",
							extend=True)

			df_field.to_csv(path_or_buf=field_csv,index_label="Parameter")
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
		file_chains=None,
		file_type="csv"
		):
		var_name = str(self.D)+"D::true"

		file_chains = self.file_chains if (file_chains is None) else file_chains
		file_out = self.dir_out+"/posterior_predictive"

		dfg = self.trace.posterior_predictive[var_name].to_dataframe().groupby("observable")
		dfs = []
		for obs,df in dfg.__iter__():
			df.reset_index("observable",drop=True,inplace=True)
			df.rename(columns={var_name:obs},inplace=True)
			dfs.append(df)
		df = pn.concat(dfs,axis=1,ignore_index=False)

		if file_type == "hdf":
			df.to_hdf(file_out+".h5",key="posterior_predictive")
		elif file_type == "csv":
			df.to_csv(file_out+".csv",index=True)
		

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

		


