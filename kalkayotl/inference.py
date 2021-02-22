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

#---------------- Matplotlib -------------------------------------
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from matplotlib import lines as mlines
#------------------------------------------------------------------

#------------ Local libraries ------------------------------------------
from kalkayotl.Models import Model1D,Model3D
from kalkayotl.Functions import AngularSeparation,CovarianceParallax,CovariancePM,get_principal,my_mode
from kalkayotl.Evidence import Evidence1D
#------------------------------------------------------------------------

class Inference:
	"""
	This class provides flexibility to infer the distance distribution given the parallax and its uncertainty
	"""
	def __init__(self,dimension,prior,parameters,
				hyper_parameters,
				dir_out,
				transformation,
				parametrization,
				zero_point,
				indep_measures=False,
				reference_system=None,
				id_name='source_id',
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
		gaia_observables = ["ra","dec","parallax","pmra","pmdec","radial_velocity",
					"ra_error","dec_error","parallax_error","pmra_error","pmdec_error","radial_velocity_error",
					"ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr",
					"dec_parallax_corr","dec_pmra_corr","dec_pmdec_corr",
					"parallax_pmra_corr","parallax_pmdec_corr",
					"pmra_pmdec_corr"]

		self.D                = dimension 
		self.prior            = prior
		self.zero_point       = zero_point
		self.parameters       = parameters
		self.hyper            = hyper_parameters
		self.dir_out          = dir_out
		self.transformation   = transformation
		self.indep_measures   = indep_measures
		self.parametrization  = parametrization
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
			

		elif self.D == 3:
			index_obs  = [0,1,2,6,7,8,12,13,16]
			index_mu   = [0,1,2]
			index_sd   = [6,7,8]
			index_corr = [12,13,16]


		elif self.D == 6:
			index_obs  = range(22)
			index_mu   = [0,1,2,3,4,5]
			index_sd   = [6,7,8,9,10,11]
			index_corr = [12,13,14,15,16,17,18,19,20,21]
			idx_plx    = 2

		else:
			sys.exit("Dimension not valid!")

		self.names_obs  = [gaia_observables[i] for i in index_obs]
		self.names_mu   = [gaia_observables[i] for i in index_mu]
		self.names_sd   = [gaia_observables[i] for i in index_sd]
		self.names_corr = [gaia_observables[i] for i in index_corr]

		self.id_name = id_name
		self.list_observables = sum([[id_name],self.names_obs],[]) 


	def load_data(self,file_data,corr_func="Lindegren+2020",*args,**kwargs):
		"""
		This function reads the data.

		Arguments:
		file_data (string): The path to a CSV file.

		id_name (string): Identifier in file_csv.

		Other arguments are passed to pandas.read_csv function

		"""

		#------- reads the data ----------------------------------------------
		data  = pn.read_csv(file_data,usecols=self.list_observables,*args,**kwargs) 
		#---------- drop na values and reorder ------------
		data  = data.dropna(thresh=len(self.list_observables))
		data  = data.reindex(columns=self.list_observables)

		#------- index as string ------
		data[self.id_name] = data[self.id_name].astype('str')

		#----- put ID as row name-----
		data.set_index(self.id_name,inplace=True)

		#----- Track ID -------------
		# In case of missing values in parallax
		IDs = []
		#----------------------------

		self.n_stars,D = np.shape(data)
		if D != 2 :
			RuntimeError("Data have incorrect shape!")

		#==================== Set Mu and Sigma =========================================
		mu_data = np.zeros(self.n_stars*self.D)
		sg_data = np.zeros((self.n_stars*self.D,self.n_stars*self.D))
		idx_tru = np.triu_indices(self.D,k=1)
		if self.D == 6:
			#----- There is no correlation with r_vel ---
			idi = np.where(idx_tru[1] != 5)[0]
			idx_tru = (idx_tru[0][idi],idx_tru[1][idi])

		for i,(ID,datum) in enumerate(data.iterrows()):
			#---Populate IDs ----------
			IDs.append(ID)
			#--------------------------
			ida  = range(i*self.D,i*self.D + self.D)
			mu   = np.array(datum[self.names_mu])  - self.zero_point
			sd   = np.array(datum[self.names_sd])
			corr = np.array(datum[self.names_corr])

			#-------- Correlation matrix of uncertainties ---------------
			rho  = np.zeros((self.D,self.D))
			rho[idx_tru] = corr
			rho  = rho + rho.T + np.eye(self.D)

			#-------- Covariance matrix of uncertainties ----------------------
			sigma = np.diag(sd).dot(rho.dot(np.diag(sd)))
			
			#---------- Insert star data --------------
			mu_data[ida] = mu
			sg_data[np.ix_(ida,ida)] = sigma
		#=========================================================================

		#----- Save identifiers ------
		df_IDs = pn.DataFrame(IDs,columns=[self.id_name])
		df_IDs.to_csv(path_or_buf=self.file_ids,index=False)
		self.ID = IDs



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
			ida_plx = [i*self.D + self.idx_plx for i in range(self.n_stars)]
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
				ida_pma = [i*self.D + self.idx_pma for i in range(self.n_stars)]
				ida_pmd = [i*self.D + self.idx_pmd for i in range(self.n_stars)]

				sg_data[np.ix_(ida_pma,ida_pma)] += cov_pms
				sg_data[np.ix_(ida_pmd,ida_pmd)] += cov_pms

		#-------- Compute inverse of covariance matrix --------------------
		self.sg_data  = sg_data
		self.tau_data = np.linalg.inv(sg_data)
		self.mu_data  = mu_data
		#=================================================================================
		print("Data correctly loaded")


	def setup(self):
		'''
		Set-up the model with the corresponding dimensions and data
		'''

		print("Configuring "+self.prior+" prior")

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
			assert self.transformation is "pc", "3D model only works in pc."

		if self.parameters["location"] is None:
			assert self.hyper["alpha"] is not None, msg_alpha

		if self.parameters["scale"] is None:
			assert self.hyper["beta"] is not None, msg_beta

		if self.prior is "EDSD":
			assert self.D == 1, "EDSD prior is only valid for 1D version."

		if self.prior is "Uniform":
			assert self.D == 1, "Uniform prior is only valid for 1D version."

		if self.prior in ["GMM","CGMM","GUM"]:
			if self.parameters["weights"] is None:
				assert self.hyper["delta"] is not None, msg_delta
			else:
				assert np.min(self.parameters["weights"])> 0.05, msg_weights

			assert self.parametrization == "central", msg_central

			if self.prior in ["CGMM","GUM"]:
				assert self.D in [3,6], "This prior is not valid for 1D version."

		if self.prior is "King":
			if self.parameters["rt"] is None:
				assert self.hyper["gamma"] is not None, msg_gamma


		if self.prior is "EFF":
			if self.parameters["gamma"] is None:
				assert self.hyper["gamma"] is not None, msg_gamma


		if self.D == 1:
			self.Model = Model1D(mu_data=self.mu_data,tau_data=self.tau_data,
								  prior=self.prior,
								  parameters=self.parameters,
								  hyper_alpha=self.hyper["alpha"],
								  hyper_beta=self.hyper["beta"],
								  hyper_gamma=self.hyper["gamma"],
								  hyper_delta=self.hyper["delta"],
								  transformation=self.transformation,
								  parametrization=self.parametrization)

		elif self.D is 3:
			self.Model = Model3D(dimension=self.D,mu_data=self.mu_data,tau_data=self.tau_data,
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




		
	def run(self,sample_iters,tuning_iters,
		chains=None,cores=None,
		step=None,
		file_chains=None,
		optimize=True,
		opt_args={
				"trials":2,
				"iterations":1000000,
				"tolerance":1e-2,
				"tolerance_type":"relative",
				"plot":True
				},
		prior_predictive=True,
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
			#-------- Prior predictive -----------------------------
			if prior_predictive:
				prior = pm.sample_prior_predictive(samples=sample_iters) #Fails for MvNorm
			else:
				prior = None
			#--------------------------------------------------------

			#---------- Posterior -----------------------
			trace = pm.sample(draws=sample_iters,
							start=start,
							step=step,
							tune=tuning_iters,
							chains=chains, cores=cores,
							progressbar=progressbar,
							discard_tuned_samples=True)

			posterior_predictive = pm.sample_posterior_predictive(trace)
			#------------------------------------------------------------

			#--------- Save with arviz ------------
			pm_data = az.from_pymc3(
						trace=trace,
						prior=prior,
						posterior_predictive=posterior_predictive)
			az.to_netcdf(pm_data,file_chains)
			#-------------------------------------


	def load_trace(self,file_chains=None):
		'''
		Loads a previously saved sampling of the model
		'''

		file_chains = self.dir_out+"/chains.nc" if (file_chains is None) else file_chains

		print("Loading existing samples ... ")
		#---------Load Trace --------------------
		
		# 
		try:
			self.ds_posterior = az.from_netcdf(file_chains).posterior
		except ValueError:
			sys.exit("There is no posterior group in {0}".format(file_chains))

		try:
			self.ds_prior = az.from_netcdf(file_chains).prior
		except:
			self.ds_prior = None

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
		rhat  = pm.stats.rhat(self.ds_posterior)
		ess   = pm.stats.ess(self.ds_posterior)

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

		print("Plotting traces ...")

		file_plots = self.dir_out+"/Traces.pdf" if (file_plots is None) else file_plots

		pdf = PdfPages(filename=file_plots)

		if IDs is not None:
			#--------- Loop over ID in list ---------------
			for i,ID in enumerate(IDs):
				id_in_IDs = np.isin(self.ID,ID)
				if not np.any(id_in_IDs) :
					sys.exit("{0} {1} is not valid".format(self.id_name,ID))
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

	def _extract(self,group="posterior",n_samples=100):
		if group == "posterior":
			data = self.ds_posterior.data_vars
		elif group == "prior":
			data = self.ds_prior.data_vars
		else:
			sys.exit("Group not recognized")
		#------------ Extract variables -----------------------------------
		locs = np.array([data[var].values for var in self.loc_variables])
		stds = np.array([data[var].values for var in self.std_variables])
		cors = np.array([data[var].values for var in self.cor_variables])
		#------------------------------------------------------------------

		#--------- Reorder indices ----------------------
		if self.prior in ["GMM"]:
			locs = np.swapaxes(locs,0,3)
		else:
			locs = np.moveaxis(locs,0,-1)[np.newaxis,:]
		#-------------------------------------------------

		#------- Take sample ---------------
		idx = np.random.choice(np.arange(locs.shape[2]),
								replace=False,
								size=n_samples)
		locs = locs[:,:,idx]
		stds = stds[:,:,idx]
		cors = cors[:,:,idx]
		#------------------------------------

		#-------- Reshape variables --------
		locs = locs.reshape((-1,3))
		stds = stds.reshape((-1,3))
		cors = cors.reshape((-1,3,3))
		#------------------------------------

		return locs,stds,cors

	def plot_model(self,
		file_plots=None,
		figsize=None,
		posterior_kwargs={"label":"Posterior",
							"color":"orange",
							"linewidth":1,
							"alpha":0.1},

		prior_kwargs={"label":"Prior",
							"color":"green",
							"linewidth":0.5,
							"alpha":0.1},

		data_kwargs={"label":"Data",
						"marker":"o",
						"color":"black",
						"size":2,
						"error_color":"grey",
						"error_lw":0.5},
		n_samples=100,
		labels=["X [pc]","Y [pc]","Z [pc]"],
		fontsize_title=16):
		"""
		This function plots the model.
		"""
		assert self.D == 3, "Only valid for 3D model ... so far"

		msg_n = "The required n_samples {0} is larger than those in the posterior.".format(n_samples)

		assert n_samples <= self.ds_posterior.sizes["draw"], msg_n

		print("Plotting model ...")

		file_plots = self.dir_out+"/Model.pdf" if (file_plots is None) else file_plots

		pdf = PdfPages(filename=file_plots)
		
		#------- Sources --------------------------------------------------
		df_source  = az.summary(self.ds_posterior,var_names=self.source_variables)

		#------------- Replace parameter id by index --------------------
		n_sources = int(df_source.shape[0]/3)
		ID  = np.repeat(np.arange(n_sources),self.D,axis=0).astype('str')
		idx = np.tile(np.arange(self.D),n_sources)

		df_source.set_index(ID,inplace=True)
		df_source.insert(loc=0,column="parameter",value=idx)

		# ------ Parameters into columns ------------------------
		suffixes  = ["_X","_Y","_Z"]
		dfs = []
		for i in range(self.D):
			idx = np.where(df_source["parameter"] == i)[0]
			tmp = df_source.drop(columns="parameter").add_suffix(suffixes[i])
			dfs.append(tmp.iloc[idx])

		#-------- Join on index --------------------
		df_source = dfs[0]
		for i,suffix in enumerate(suffixes[1:]):
			df_source = df_source.join(dfs[i+1],
				how="inner",lsuffix="",rsuffix=suffix)

		srcs_loc = df_source[["mean_X","mean_Y","mean_Z"]].to_numpy()
		srcs_std = df_source[["sd_X","sd_Y","sd_Z"]].to_numpy()
		#-------------------------------------------------------------------

		#---------- Extract prior and posterior -----------------
		pos_locs,pos_stds,pos_corrs = self._extract(group="posterior",n_samples=n_samples)
		if self.ds_prior is not None:
			pri_locs,pri_stds,pri_corrs = self._extract(group="prior",n_samples=n_samples)

		fig, axs = plt.subplots(nrows=2,ncols=2,figsize=figsize)
		for ax,idx in zip([axs[0,0],axs[0,1],axs[1,0]],[[0,1],[2,1],[0,2]]):
			#--------- Sources --------------------------
			ax.errorbar(x=srcs_loc[:,idx[0]],
						y=srcs_loc[:,idx[1]],
						xerr=srcs_std[:,idx[0]],
						yerr=srcs_std[:,idx[1]],
						fmt='none',
						ecolor=data_kwargs["error_color"],
						elinewidth=data_kwargs["error_lw"],
						zorder=1)
			ax.scatter(x=srcs_loc[:,idx[0]],
						y=srcs_loc[:,idx[1]],
						marker=data_kwargs["marker"],
						color=data_kwargs["color"],
						s=data_kwargs["size"],
						zorder=1)

			#-------- Posterior ----------------------------------------------------------
			for mu,std,corr in zip(pos_locs,pos_stds,pos_corrs):
					width, height, angle = get_principal(std,corr,idx)
					ell  = Ellipse(mu[idx],width=width,height=height,angle=angle,
									clip_box=ax.bbox,
									edgecolor=posterior_kwargs["color"],
									facecolor=None,
									fill=False,
									linewidth=posterior_kwargs["linewidth"],
									alpha=posterior_kwargs["alpha"],
									zorder=2)
					ax.add_artist(ell)
			#-----------------------------------------------------------------------------

			#-------- Prior ----------------------------------------------------------
			if self.ds_prior is not None:
				for mu,std,corr in zip(pri_locs,pri_stds,pri_corrs):
						width, height, angle = get_principal(std,corr,idx)
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

		#------------- Legend -----------------------------------------------------------
		prior_line = mlines.Line2D([], [], color=prior_kwargs["color"], 
								marker=None, label=prior_kwargs["label"])
		posterior_line = mlines.Line2D([], [], color=posterior_kwargs["color"], 
								marker=None, label=posterior_kwargs["label"])
		data_mrkr =  mlines.Line2D([], [], marker=data_kwargs["marker"], color="w", 
						  markerfacecolor=data_kwargs["color"], 
						  markersize=5,
						  label=data_kwargs["label"])
		if self.ds_prior is not None:
			handles = [prior_line,posterior_line,data_mrkr]
		else:
			handles = [posterior_line,data_mrkr]
		axs[1,1].legend(handles=handles,loc='center')
		axs[1,1].axis("off")
		#-------------------------------------------------------------------------------

		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)
		pdf.savefig(bbox_inches='tight')
		plt.close()
		#---------------------------------------------------------------------------------------------
		pdf.close()


	def save_statistics(self,hdi_prob=0.95):
		'''
		Saves the statistics to a csv file.
		Arguments:
		
		'''
		print("Computing statistics ...")

		#----------------------- Functions ---------------------------------
		stat_funcs = {"median":lambda x:np.median(x),
					  "mode":lambda x:my_mode(x)}
		#---------------------------------------------------------------------

		#------- Get IDs -----------------------
		IDs = pn.read_csv(self.file_ids)[self.id_name].values
		#---------------------------------------

		#-------------- Source statistics ----------------------------------------------------
		source_csv = self.dir_out +"/Sources_statistics.csv"
		df_source  = az.summary(self.ds_posterior,
						var_names=self.source_variables,
						stat_funcs=stat_funcs,
						hdi_prob=hdi_prob,
						extend=True)

		#------------- Replace parameter id by source ID--------------------
		n_sources = len(IDs)
		ID  = np.repeat(IDs,self.D,axis=0)
		idx = np.tile(np.arange(self.D),n_sources)

		df_source.set_index(ID,inplace=True)
		df_source.insert(loc=0,column="parameter",value=idx)

		if self.D == 3 :
			# ------ Parameters into columns ------------------------
			suffixes  = ["_X","_Y","_Z"]
			dfs = []
			for i in range(self.D):
				idx = np.where(df_source["parameter"] == i)[0]
				tmp = df_source.drop(columns="parameter").add_suffix(suffixes[i])
				dfs.append(tmp.iloc[idx])

			#-------- Join on index --------------------
			df_source = dfs[0]
			for i,suffix in enumerate(suffixes[1:]):
				df_source = df_source.join(dfs[i+1],
					how="inner",lsuffix="",rsuffix=suffix)
			#---------------------------------------------------------------------

		#---------- Save source data frame ----------------------
		df_source.to_csv(path_or_buf=source_csv,index_label=self.id_name)

		#-------------- Global statistics ------------------------
		if len(self.cluster_variables) > 0:
			global_csv = self.dir_out +"/Cluster_statistics.csv"
			df_global = az.summary(self.ds_posterior,var_names=self.stats_variables,
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
				hyper_alpha=self.hyper_alpha,
				hyper_beta=self.hyper_beta,
				hyper_gamma=self.hyper_gamma,
				hyper_delta=self.hyper_delta,
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

		


