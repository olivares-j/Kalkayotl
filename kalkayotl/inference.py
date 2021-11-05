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
import pymc3 as pm
import numpy as np
import pandas as pn
import arviz as az
import h5py
import scipy.stats as st
from scipy.linalg import inv as inverse
from string import ascii_uppercase

#---------------- Matplotlib -------------------------------------
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from matplotlib.colors import TwoSlopeNorm,Normalize
from matplotlib import lines as mlines
#------------------------------------------------------------------

#------------ Local libraries ------------------------------------------
from kalkayotl.Models import Model1D,Model3D,Model6D
from kalkayotl.Functions import AngularSeparation,CovarianceParallax,CovariancePM,get_principal,my_mode
# from kalkayotl.Evidence import Evidence1D
from kalkayotl.Transformations import astrometryToPhaseSpace
#------------------------------------------------------------------------

class Inference:
	"""
	This class provides flexibility to infer the distance distribution given the parallax and its uncertainty
	"""
	def __init__(self,dimension,
				dir_out,
				zero_point,
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
		self.suffixes  = ["_X","_Y","_Z","_U","_V","_W"]

		self.D                = dimension 
		self.zero_point       = zero_point
		self.dir_out          = dir_out
		self.indep_measures   = indep_measures
		
		self.reference_system = reference_system
		self.file_ids         = self.dir_out+"/Identifiers.csv"

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

		self.id_name = id_name
		self.list_observables = sum([[id_name],self.names_obs],[]) 


	def load_data(self,file_data,corr_func="Lindegren+2020",*args,**kwargs):
		"""
		This function reads the data.

		Arguments:
		file_data (string): The path to a CSV file.

		corr_func (string): Type of angular correlation.

		Other arguments are passed to pandas.read_csv function

		"""

		#------- Reads the data ---------------------------------------------------
		data  = pn.read_csv(file_data,usecols=self.list_observables,*args,**kwargs) 

		#---------- Order ----------------------------------
		data  = data.reindex(columns=self.list_observables)

		#------- ID as string ----------------------------
		data[self.id_name] = data[self.id_name].astype('str')

		#----- ID as index ----------------------
		data.set_index(self.id_name,inplace=True)

		#-------- Drop NaNs ------------------------
		data.dropna(subset=self.names_nan,inplace=True,
						thresh=len(self.names_nan))
		#----------------------------------------------

		#--------- Mean values ---------------------------------
		self.mean_astrometry = data[self.names_mu].mean().values
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
			mu   = np.array(datum[self.names_mu]) - self.zero_point
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
				transformation):
		'''
		Set-up the model with the corresponding dimensions and data
		'''

		self.prior            = prior
		self.parameters       = parameters
		self.hyper            = hyper_parameters
		self.parametrization  = parametrization
		self.transformation   = transformation

		print(15*"+", " Prior setup ", 15*"+")
		print("Type of prior: ",self.prior)

		msg_alpha = "hyper_alpha must be specified."
		msg_beta  = "hyper_beta must be specified."
		msg_trans = "Transformation must be either pc or mas."
		msg_delta = "hyper_delta must be specified."
		msg_gamma = "hyper_gamma must be specified."
		msg_central = "Only the central parametrization is valid for this configuration."
		msg_non_central = "Only the non-central parametrization is valid for this configuration."
		msg_weights = "weights must be greater than 5%."

		assert self.transformation in ["pc","mas"], msg_trans

		if self.D in [3,6]:
			assert self.transformation == "pc", "3D model only works in pc."

		#============== Mixtures =====================================================
		if self.prior in ["GMM","CGMM","GUM"]:

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

			if self.prior in ["CGMM","GUM"]:
				assert self.D in [3,6], "This prior is not valid for 1D version."
		#============================================================================

		#====================== Location ==========================================================
		if self.parameters["location"] is None:
			#---------------- Alpha ---------------------------------------------------------
			if self.hyper["alpha"] is None:

				#-- Cluster dispersion ----
				xyz_sd,uvw_sd = 10.,10.
				#-------------------------

				#------------------------ Cluster mean value ------------------------
				if self.D == 6:
					#------------ Cluster mean coordinates -------------------
					x,y,z,u,v,w = astrometryToPhaseSpace(
									self.mean_astrometry[np.newaxis,:],
									reference_system=self.reference_system)[0]
					#----------------------------------------------------------

					self.hyper["alpha"] = [[x,xyz_sd],[y,xyz_sd],[z,xyz_sd],
										   [u,uvw_sd],[v,uvw_sd],[w,uvw_sd]]
				elif self.D == 3:
					#------------ Cluster mean coordinates -------------------
					x,y,z,_,_,_ = astrometryToPhaseSpace(np.append(
									self.mean_astrometry,[0.0,0.0,0.0])[np.newaxis,:],
									reference_system=self.reference_system)[0]
					#----------------------------------------------------------

					self.hyper["alpha"] = [[x,xyz_sd],[y,xyz_sd],[z,xyz_sd]]
				#----------------------------------------------------------------------


				print("The alpha hyper-parameter has been set to:")
				names = ["X","Y","Z","U","V","W"]
				for i,value in enumerate(self.hyper["alpha"]):
					print("{0}: {1:2.1f} +/- {2:2.1f}".format(names[i],value[0],value[1]))
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
				if self.prior in ["GMM","CGMM","GUM"]:
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
			if self.prior in ["GMM","CGMM","GUM"]:
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
				self.hyper["beta"] = 10.0
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
				if self.prior in ["GMM","CGMM","GUM"]:
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
			if self.prior in ["GMM","CGMM","GUM"]:
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

		

		if self.parameters["scale"] is None and self.hyper["eta"] is None:
			self.hyper["eta"] = 10.0
			print("The eta hyper-parameter has been set to:")
			print(self.hyper["eta"])

		if self.prior == "EDSD":
			assert self.D == 1, "EDSD prior is only valid for 1D version."

		if self.prior == "Uniform":
			assert self.D == 1, "Uniform prior is only valid for 1D version."

		if self.prior == "King":
			if self.parameters["rt"] is None:
				assert self.hyper["gamma"] is not None, msg_gamma


		if self.prior == "EFF":
			if self.parameters["gamma"] is None:
				assert self.hyper["gamma"] is not None, msg_gamma


		if self.D == 1:
			self.Model = Model1D(n_sources=self.n_sources,
								mu_data=self.mu_data,
								tau_data=self.tau_data,
								prior=self.prior,
								parameters=self.parameters,
								hyper_alpha=self.hyper["alpha"],
								hyper_beta=self.hyper["beta"],
								hyper_gamma=self.hyper["gamma"],
								hyper_delta=self.hyper["delta"],
								transformation=self.transformation,
								parametrization=self.parametrization)

		elif self.D == 3:
			self.Model = Model3D(n_sources=self.n_sources,
								mu_data=self.mu_data,
								tau_data=self.tau_data,
								prior=self.prior,
								parameters=self.parameters,
								hyper_alpha=self.hyper["alpha"],
								hyper_beta=self.hyper["beta"],
								hyper_gamma=self.hyper["gamma"],
								hyper_delta=self.hyper["delta"],
								hyper_eta=self.hyper["eta"],
								transformation=self.transformation,
								reference_system=self.reference_system,
								parametrization=self.parametrization)
		elif self.D == 6:
			self.Model = Model6D(n_sources=self.n_sources,
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
								transformation=self.transformation,
								reference_system=self.reference_system,
								parametrization=self.parametrization)
		else:
			sys.exit("Dimension not valid!")

		print((30+13)*"+")




		
	def run(self,sample_iters,tuning_iters,
		chains=None,cores=None,
		step=None,
		file_chains=None,
		optimize=True,
		opt_args={
				"trials":1,
				"iterations":100000
				"tolerance":1e-2,
				"tolerance_type":"relative",
				"plot":True
				},
		prior_predictive=False,
		posterior_predictive=False,
		progressbar=True,
		*args,**kwargs):
		"""
		Performs the MCMC run.
		Arguments:
		sample_iters (integer):    Number of MCMC iterations.
		tuning_iters (integer):    Number of burning iterations.
		"""

		file_chains = self.dir_out+"/chains.nc" if (file_chains is None) else file_chains

		#-------------- ADVI+ADAPT_DIAG ----------------------------------------------------------
		if optimize:
			print("Using advi+adapt_diag to optimize the initial solution ...")
			trials = []
			min_trials = []

			for i in range(opt_args["trials"]):
				print("Trial {0}".format(i+1))
				approx = pm.fit(
					random_seed=None,
					n=opt_args["iterations"],
					method="advi",
					model=self.Model,
					callbacks=[pm.callbacks.CheckParametersConvergence(
								tolerance=opt_args["tolerance"], 
								diff=opt_args["tolerance_type"])],
					progressbar=True,
					obj_optimizer=pm.adagrad_window)
				trials.append(approx)
				min_trials.append(np.min(approx.hist))

			#-------- Best one -----------------
			best = trials[np.argmin(min_trials)]
			#-----------------------------------
			
			#------------- Plot trials ----------------------------------
			if opt_args["plot"]:
				plt.figure()
				for i,app in enumerate(trials):
					plt.plot(app.hist,label="Trial {0}".format(i+1))
				plt.plot(best.hist,label="Best one")
				plt.legend()
				plt.xlabel("Iterations")
				plt.ylabel("Average Loss")
				plt.savefig(self.dir_out+"/Initializations.png")
				plt.close()
			#-----------------------------------------------------------

			#----------- Mean field approximation ------------------------------------
			start = best.sample(draws=chains)
			start = list(start)
			stds = best.bij.rmap(best.std.eval())
			cov = self.Model.dict_to_array(stds) ** 2
			mean = best.bij.rmap(best.mean.get_value())
			mean = self.Model.dict_to_array(mean)
			weight = 50
			potential = pm.step_methods.hmc.quadpotential.QuadPotentialDiagAdapt(
												self.Model.ndim, mean, cov, weight)
			step = pm.NUTS(potential=potential, model=self.Model, **kwargs)
			#------------------------------------------------------------------------
		else:
			start = None
			step = None

		print("Sampling the model ...")

		with self.Model:
			#-------- Prior predictive ----------------------------------
			if prior_predictive:
				prior = pm.sample_prior_predictive(samples=sample_iters) #Fails for MvNorm
			else:
				prior = None
			#-------------------------------------------------------------

			#---------- Posterior -----------------------
			trace = pm.sample(draws=sample_iters,
							start=start,
							step=step,
							tune=tuning_iters,
							chains=chains, cores=cores,
							progressbar=progressbar,
							discard_tuned_samples=True,
							return_inferencedata=False)
			#-----------------------------------------------

			#-------- Posterior predictive -----------------------------
			if posterior_predictive:
				predictive = pm.sample_posterior_predictive(trace)
			else:
				predictive = None
			#--------------------------------------------------------

			#--------- Save with arviz ------------
			pm_data = az.from_pymc3(
						trace=trace,
						prior=prior,
						posterior_predictive=predictive)
			az.to_netcdf(pm_data,file_chains)
			#-------------------------------------


	def load_trace(self,file_chains=None):
		'''
		Loads a previously saved sampling of the model
		'''

		file_chains = self.dir_out+"/chains.nc" if (file_chains is None) else file_chains

		if not hasattr(self,"ID"):
			#----- Load identifiers ------
			self.ID = pn.read_csv(self.file_ids).to_numpy().flatten()

		print("Loading existing samples ... ")
		#---------Load Trace ---------------------------------------------------
		try:
			self.ds_posterior = az.from_netcdf(file_chains).posterior
		except ValueError:
			sys.exit("There is no posterior group in {0}".format(file_chains))
		#------------------------------------------------------------------------

		#----------- Load prior -------------------------------------------------
		try:
			self.ds_prior = az.from_netcdf(file_chains).prior
		except:
			self.ds_prior = None
		#-------------------------------------------------------------------------

		#------- Variable names -----------------------------------------------------------
		source_variables = list(filter(lambda x: "source" in x, self.ds_posterior.data_vars))
		cluster_variables = list(filter(lambda x: ( ("loc" in x) 
											or ("scl" in x) 
											or ("weights" in x)
											or ("beta" in x)
											or ("gamma" in x)
											or ("rt" in x)),self.ds_posterior.data_vars))
	
		plots_variables = cluster_variables.copy()
		stats_variables = cluster_variables.copy()
		cluster_loc_var = cluster_variables.copy()
		cluster_std_var = cluster_variables.copy()
		cluster_cor_var = cluster_variables.copy()

		#----------- Case specific variables -------------
		tmp_plots = cluster_variables.copy()
		tmp_stats = cluster_variables.copy()
		tmp_loc   = cluster_variables.copy()
		tmp_stds  = cluster_variables.copy()
		tmp_corr  = cluster_variables.copy()

		if self.D in [3,6]:
			for var in tmp_plots:
				if "scl" in var and "stds" not in var:
					plots_variables.remove(var)

			for var in tmp_stats:
				if "scl" in var:
					if not ("stds" in var or "corr" in var or "unif" in var):
						stats_variables.remove(var)

			for var in tmp_loc:
				if "loc" not in var:
					cluster_loc_var.remove(var)

			for var in tmp_stds:
				if "stds" not in var:
					cluster_std_var.remove(var)

			for var in tmp_corr:
				if "corr" not in var:
					cluster_cor_var.remove(var)
		#----------------------------------------------------

		self.source_variables  = source_variables
		self.cluster_variables = cluster_variables
		self.plots_variables   = plots_variables
		self.stats_variables   = stats_variables
		self.loc_variables     = cluster_loc_var
		self.std_variables     = cluster_std_var
		self.cor_variables     = cluster_cor_var

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
				coords = {str(self.D)+"D_source_dim_0" : idx}
				plt.figure(0)
				axes = az.plot_trace(self.ds_posterior,
						var_names=self.source_variables,
						coords=coords,
						figsize=figsize,
						lines=lines, 
						combined=combined, 
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

		if len(self.cluster_variables) > 0:
			plt.figure(1)
			axes = az.plot_trace(self.ds_posterior,
					var_names=self.plots_variables,
					figsize=figsize,
					lines=lines, 
					combined=combined,
					plot_kwargs=plot_kwargs, 
					hist_kwargs=hist_kwargs, 
					trace_kwargs=trace_kwargs)

			for ax in axes:
				# --- Set units in parameters ------------------------------
				title = ax[0].get_title()
				if ("loc" in title) or ("scl" in title):
					if self.transformation == "pc":
						ax[0].set_xlabel("pc")
					else:
						ax[0].set_xlabel("mas")
					#-----------------------------------------------------------
				ax[1].set_xlabel("Iteration")

			plt.gcf().suptitle("Population parameters",fontsize=fontsize_title)
				
			#-------------- Save fig --------------------------
			pdf.savefig(bbox_inches='tight')
			plt.close(1)

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

		#------ Organize sources ---
		srcs = srcs.squeeze(axis=0)
		srcs = np.moveaxis(srcs,2,0)
		#---------------------------

		#------ Dimensions -----
		n,nc,ns,nd = srcs.shape
		#-----------------------

		#--- One or multiple chains -----------
		if chain is None:
			#-------- Merge chains ----------
			srcs = srcs.reshape((n,nc*ns,nd))
			#--------------------------------

		else:
			#--- Extract chain -------
			srcs = srcs[:,chain]
			#-------------------------
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
			stds = np.array([data[var].values for var in self.std_variables])
		#--------------------------------------------------------------------

		#----------- Extract correlations -------------------------------
		if len(self.std_variables) == 0:
			cors = np.array(self.parameters["corr"])
			cors = cors[:,np.newaxis,np.newaxis,:]
			cors = np.tile(cors,(1,nc,ns,1,1))
		else:
			cors = np.array([data[var].values for var in self.cor_variables])
		#------------------------------------------------------------------

		
		#--------- Reorder indices ----------------------
		if self.prior in ["GMM","CGMM"]:
			if str(self.D)+"D_weights" in self.cluster_variables:
				amps = np.array(data[str(self.D)+"D_weights"].values)
				amps = np.moveaxis(amps,2,0)
			else:
				amps = np.array(self.parameters["weights"])
				amps = amps[:,np.newaxis,np.newaxis]
				amps = np.tile(amps,(1,nc,ns))

			if self.prior == "GMM":
				locs = np.swapaxes(locs,0,3)
			else:
				locs = np.moveaxis(locs,0,-1)[np.newaxis,:]
				locs = np.tile(locs,(stds.shape[0],1,1,1))

		else:
			locs = np.moveaxis(locs,0,-1)[np.newaxis,:]
			amps = np.ones_like(locs)[:,:,:,0]
		#-------------------------------------------------

		
		#---------- One or multiple chains -------
		if chain is None:
			#-------- Merge chains --------------
			ng,nc,ns,nd = locs.shape
			amps = amps.reshape((ng,nc*ns))
			locs = locs.reshape((ng,nc*ns,nd))
			stds = stds.reshape((ng,nc*ns,nd))
			cors = cors.reshape((ng,nc*ns,nd,nd))
			#------------------------------------

		else:
			#--- Extract chain --
			amps = amps[:,chain]
			locs = locs[:,chain]
			stds = stds[:,chain]
			cors = cors[:,chain]
			#--------------------
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

	def _classify(self,srcs,amps,locs,covs):
		'''
		Obtain the class of each source at each chain step
		'''
		print("Classifying sources ...")

		if self.prior in ["GMM","CGMM"]:
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

			grps = st.mode(log_lk.argmax(axis=2),axis=1)[0].flatten()

		else:
			grps = np.zeros(len(self.ID))

		self.df_groups = pn.DataFrame(data={"group":grps},index=self.ID)

	def _kinematic_indices(self,group="posterior",chain=None,n_samples=None):
		'''
		Compute the kinematic indicators of expansion and rotation
		'''
		#---- Get parameters -------
		srcs,_,locs,_ = self._extract(group=group,
										n_samples=n_samples,
										chain=chain)

		if self.prior in ["GMM","CGMM"]:
			sys.exit("Kinematic indices are not available for mixture models!")


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

		#---- Normalized position -----------------------
		nrs = np.linalg.norm(rs,axis=2,keepdims=True)
		ers = rs/nrs
		#-----------------------------------------------

		#------- Products ---------------
		exp = np.empty((ers.shape[0],ers.shape[1]))
		rot = np.empty((ers.shape[0],ers.shape[1]))
		for i,(er, v) in enumerate(zip(ers,vs)):
			exp[i] = np.diag(np.inner(er,v))
			rot[i] = np.linalg.norm(np.cross(er,v),axis=1)
		
		return nrs.squeeze(),exp,rot

	def plot_model(self,
		file_plots=None,
		figsize=None,
		n_samples=100,
		chain=None,
		fontsize_title=16,
		labels=["X [pc]","Y [pc]","Z [pc]",
				"U [km/s]","V [km/s]","W [km/s]"],
		group_kwargs={"label":"Group",
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
						"cmap_mix":"tab10",
						"cmap_pos":"coolwarm",
						"cmap_vel":"summer"},
		source_labels={i:v for i,v in enumerate(ascii_uppercase)},
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
		if self.prior in ["GMM","CGMM"]:
			chain = 0 if chain is None else chain
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
			self._classify(pos_srcs,pos_amps,pos_locs,pos_covs)
		#------------------------------------------------

		#-- Sources mean and standard deviation ---------
		srcs_loc = np.mean(pos_srcs,axis=1)
		srcs_std = np.std(pos_srcs,axis=1)
		#------------------------------------------------

		#======================== Colors ================================
		if self.prior in ["GMM","CGMM"] or self.D == 3:
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
			nrs, exp, rot = self._kinematic_indices(group="posterior")
			print("Expansion: {0:2.1f} +/- {1:2.1f} km/s".format(
											np.mean(exp),np.std(exp)))
			print("Rotation:  {0:2.1f} +/- {1:2.1f} km/s".format(
											np.mean(rot),np.std(rot)))
			exp = np.mean(exp,axis=1)
			nrs = np.mean(nrs,axis=1)
			#----------------------------------------------------------

			#------------ Colormaps --------------------------
			cmap_pos = cm.get_cmap(source_kwargs["cmap_pos"])
			cmap_vel = cm.get_cmap(source_kwargs["cmap_vel"])
			#-------------------------------------------------

			#------------ Normalizations ------------------------
			norm_pos = TwoSlopeNorm(vcenter=0,
								vmin=exp.min(),vmax=exp.max())
			norm_vel = Normalize(vmin=nrs.min(),vmax=nrs.max())
			#----------------------------------------------------

			#--------- Sources colors ------------
			srcs_clr_pos = cmap_pos(norm_pos(exp))
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
						zorder=1)
			ax.scatter(x=srcs_loc[:,idx[0]],
						y=srcs_loc[:,idx[1]],
						c=srcs_clr_pos,
						marker=source_kwargs["marker"],
						s=source_kwargs["size"],
						zorder=1)

			#-------- Posterior ----------------------------------------------------------
			for mus,covs in zip(pos_locs,pos_covs):
				for mu,cov in zip(mus,covs):
						width, height, angle = get_principal(cov,idx)
						ell  = Ellipse(mu[idx],width=width,height=height,angle=angle,
										clip_box=ax.bbox,
										edgecolor=group_kwargs["color"],
										facecolor=None,
										fill=False,
										linewidth=group_kwargs["linewidth"],
										alpha=group_kwargs["alpha"],
										zorder=2)
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

			#------------- Titles -------------------------------------
			ax.set_xlabel(labels[idx[0]])
			ax.set_ylabel(labels[idx[1]])

		axs[0,0].axes.xaxis.set_visible(False)
		axs[0,1].axes.yaxis.set_visible(False)

		#------------- Legend lines  ---------------------------------------
		prior_line = mlines.Line2D([], [], color=prior_kwargs["color"], 
								marker=None, label=prior_kwargs["label"])
		group_line = mlines.Line2D([], [], color=group_kwargs["color"], 
								marker=None, label=group_kwargs["label"])
		#-------------------------------------------------------------------

		#----------- Legend symbols ----------------------------------
		if self.prior in ["GMM","CGMM"]:
			source_mrkr =  [mlines.Line2D([], [], 
								marker=source_kwargs["marker"], color="w", 
								markerfacecolor=cmap_pos(norm_pos(g)), 
								markersize=5,
								label=source_labels[g]) 
								for g in np.unique(groups)] 
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
		if self.prior not in ["GMM","CGMM"] and self.D == 6:
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
							zorder=1)
				clr_vel = ax.scatter(x=srcs_loc[:,idx[0]],
							y=srcs_loc[:,idx[1]],
							c=srcs_clr_vel,
							marker=source_kwargs["marker"],
							s=source_kwargs["size"],
							zorder=1)

				#-------- Posterior ----------------------------------------------------------
				for mus,covs in zip(pos_locs,pos_covs):
					for mu,cov in zip(mus,covs):
							width, height, angle = get_principal(cov,idx)
							ell  = Ellipse(mu[idx],width=width,height=height,angle=angle,
											clip_box=ax.bbox,
											edgecolor=group_kwargs["color"],
											facecolor=None,
											fill=False,
											linewidth=group_kwargs["linewidth"],
											alpha=group_kwargs["alpha"],
											zorder=2)
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

			axs[0,0].axes.xaxis.set_visible(False)
			axs[0,1].axes.yaxis.set_visible(False)

			#------------- Legend lines --------------------------------------
			prior_line = mlines.Line2D([], [], color=prior_kwargs["color"], 
									marker=None, label=prior_kwargs["label"])
			group_line = mlines.Line2D([], [], color=group_kwargs["color"], 
									marker=None, label=group_kwargs["label"])
			#-----------------------------------------------------------------

			#----------- Legend symbols ----------------------------------
			if self.prior in ["GMM","CGMM"]:
				source_mrkr = [mlines.Line2D([], [], marker=source_kwargs["marker"], 
							  	color="w", 
							  	markerfacecolor=cmap_vel(norm_vel(g)), 
							  	markersize=5,
							  	label=source_labels[g]) 
								for g in np.unique(groups)] 
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
			if self.prior not in ["GMM","CGMM"]:
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


	def save_statistics(self,hdi_prob=0.95,chain_gmm=[0]):
		'''
		Saves the statistics to a csv file.
		Arguments:
		
		'''
		print("Computing statistics ...")

		#----------------------- Functions ---------------------------------
		stat_funcs = {"median":lambda x:np.median(x),
					  "mode":lambda x:my_mode(x)}
		def distance(x,y,z):
			return np.sqrt(x**2 + y**2 + z**2)
		#---------------------------------------------------------------------
		
		#--------- Coordinates -------------------------
		# In MM use only one chain
		if self.prior in ["GMM","CGMM"]:
			print("WARNING: In mixture models only one "\
				+"chain is used to compute statistics.\n"\
				+"Set chain_gmm=[0,1,..,n_chains] to override.")
			data = az.utils.get_coords(self.ds_posterior,{"chain":chain_gmm})
		else:
			data = self.ds_posterior
		#------------------------------------------------------------

		#-------------- Source statistics ----------------------------------------------------
		source_csv = self.dir_out +"/Sources_statistics.csv"
		df_source  = az.summary(data,var_names=self.source_variables,
						stat_funcs=stat_funcs,
						hdi_prob=hdi_prob,
						extend=True)

		#------------- Replace parameter id by source ID--------------------
		n_sources = len(self.ID)
		ID  = np.repeat(self.ID,self.D,axis=0)
		idx = np.tile(np.arange(self.D),n_sources)

		df_source.set_index(ID,inplace=True)
		df_source.insert(loc=0,column="parameter",value=idx)
		#---------------------------------------------------------------

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
			self._classify(pos_srcs,pos_amps,pos_locs,pos_covs)
		#------------------------------------------------

		if self.D in [3,6] :
			# ------ Parameters into columns ------------------------
			dfs = []
			for i in range(self.D):
				idx = np.where(df_source["parameter"] == i)[0]
				tmp = df_source.drop(columns="parameter").add_suffix(self.suffixes[i])
				dfs.append(tmp.iloc[idx])

			#-------- Join on index --------------------
			df_source = dfs[0]
			for i in range(1,self.D) :
				df_source = df_source.join(dfs[i],
					how="inner",lsuffix="",rsuffix=self.suffixes[i])
			#---------------------------------------------------------------------

		#---------- Add group -----------------------------------
		df_source = df_source.join(self.df_groups)
		#----------------------------------------------

		#------ Add distance ---------------------------------------------------------
		df_source["mode_distance"] = df_source[["mode_X","mode_Y","mode_Z"]].apply(
			lambda x: distance(*x),axis=1)

		df_source["mean_distance"] = df_source[["mean_X","mean_Y","mean_Z"]].apply(
			lambda x: distance(*x),axis=1)

		df_source["median_distance"] = df_source[["median_X","median_Y","median_Z"]].apply(
			lambda x: distance(*x),axis=1)
		#----------------------------------------------------------------------------

		#---------- Save source data frame ----------------------
		df_source.to_csv(path_or_buf=source_csv,index_label=self.id_name)

		#-------------- Global statistics ------------------------
		if len(self.cluster_variables) > 0:
			global_csv = self.dir_out +"/Cluster_statistics.csv"
			df_global = az.summary(data,var_names=self.stats_variables,
							stat_funcs=stat_funcs,
							hdi_prob=hdi_prob,
							extend=True)

			df_global.to_csv(path_or_buf=global_csv,index_label="Parameter")

	def save_samples(self,merge=True):
		'''
		Saves the chain samples to an h5 file.
		Arguments:
		dir_csv (string) Directory where to save the samples
		'''
		print("Saving samples ...")

		#------- Get IDs -----------------------
		IDs = pn.read_csv(self.file_ids)[self.id_name].values.astype('str')
		#---------------------------------------

		#------ Open h5 file -------------------
		file_h5 = self.dir_out + "/Samples.h5"

		sources_trace = self.ds_posterior[self.source_variables].to_array().T

		with h5py.File(file_h5,'w') as hf:
			grp_glb = hf.create_group("Cluster")
			grp_src = hf.create_group("Sources")

			#------ Loop over global parameters ---
			for name in self.cluster_variables:
				data = np.array(self.ds_posterior[name]).T
				if merge:
					data = data.reshape((data.shape[0],-1))
				grp_glb.create_dataset(name, data=data)

			#------ Loop over source parameters ---
			for i,name in enumerate(IDs):
				data = sources_trace[{str(self.D)+"D_source_dim_0" : i}].values
				if merge:
					data = data.reshape((data.shape[0],-1))
				grp_src.create_dataset(name, data=data)


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

		


