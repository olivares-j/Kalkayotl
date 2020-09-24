'''
Copyright 2019 Javier Olivares Romero

This file is part of Kalkayotl.

    Kalkayotl is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyAspidistra is distributed in the hope that it will be useful,
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
import h5py
import scipy.stats as st
from scipy.linalg import inv as inverse

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#----- For GMM experimental initializer ---------
from pymc3.step_methods.hmc import quadpotential

#------------ Local libraries ------------------------------------------
from kalkayotl.Models import Model1D,ModelND
from kalkayotl.Functions import AngularSeparation,CovarianceParallax,CovariancePM
from kalkayotl.Evidence import Evidence1D
#------------------------------------------------------------------------

class Inference:
	"""
	This class provides flexibility to infer the distance distribution given the parallax and its uncertainty
	"""
	def __init__(self,dimension,prior,parameters,
				hyper_alpha,
				hyper_beta,
				hyper_gamma,
				hyper_delta,
				dir_out,
				transformation,
				parametrization,
				zero_point,
				indep_measures=False,
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
		self.hyper_alpha      = hyper_alpha
		self.hyper_beta       = hyper_beta
		self.hyper_gamma      = hyper_gamma
		self.hyper_delta      = hyper_delta
		self.dir_out          = dir_out
		self.transformation   = transformation
		self.indep_measures   = indep_measures
		self.parametrization  = parametrization

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


		elif self.D == 5:
			index_obs  = [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21]
			index_mu   = [0,1,2,3,4]
			index_sd   = [6,7,8,9,10]
			index_corr = [12,13,14,15,16,17,18,19,20,21]
			idx_plx    = 2


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


		if prior is "Gaussian":
			assert hyper_delta is None, "Parameter hyper_delta is only valid for GMM prior."


	def load_data(self,file_data,id_name='source_id',radec_inflation=10.0,id_length=10,corr_func="Vasiliev+2019",*args,**kwargs):
		"""
		This function reads the data.

		Arguments:
		file_data (string): The path to a CSV file.

		id_name (string): Identifier in file_csv.

		Other arguments are passed to pandas.read_csv function

		"""
		self.id_name = id_name
		list_observables = sum([[id_name],self.names_obs],[])

		#------- reads the data ----------------------------------------------
		data  = pn.read_csv(file_data,usecols=list_observables,*args,**kwargs) 
		#---------- drop na values and reorder ------------
		data  = data.dropna(thresh=len(list_observables))
		data  = data.reindex(columns=list_observables)

		#------- index as string ------
		data[list_observables[0]] = data[list_observables[0]].astype('str')

		if self.D > 1:
			#--- Uncertainties in must be in same units as means -----------
			data["ra_error"]       = data["ra_error"]/(1e3*3600.0)  + radec_inflation/3600.0
			data["dec_error"]      = data["dec_error"]/(1e3*3600.0) + radec_inflation/3600.0
		#============================================================

		#----- put ID as row name-----
		data.set_index(list_observables[0],inplace=True)

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
		df_IDs = pn.DataFrame(IDs,columns=["ID"])
		df_IDs.to_csv(path_or_buf=self.dir_out+"/Identifiers.csv",index_label="Parameter")
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
			
			if self.D > 3:
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

		if self.parameters["location"] is None:
			assert self.hyper_alpha is not None, "hyper_alpha must be specified."

		if self.parameters["scale"] is None:
			assert self.hyper_beta is not None, "hyper_beta must be specified."

		assert self.transformation in ["pc","mas"],"Transformation must be either pc or mas"

		if self.prior is "GMM":
			if self.parameters["weights"] is None:
				assert self.hyper_delta is not None, "hyper_delta must be specified."
			else:
				assert np.min(self.parameters["weights"])> 0.05, "weights must be greater than 5%"

		if self.prior is "King":
			if self.parameters["rt"] is None:
				assert self.hyper_gamma is not None, "hyper_gamma must be specified."

		if self.prior is "EFF":
			if self.parameters["gamma"] is None:
				assert self.hyper_gamma is not None, "hyper_gamma must be specified."


		if self.D == 1:
			self.Model = Model1D(mu_data=self.mu_data,tau_data=self.tau_data,
								  prior=self.prior,
								  parameters=self.parameters,
								  hyper_alpha=self.hyper_alpha,
								  hyper_beta=self.hyper_beta,
								  hyper_gamma=self.hyper_gamma,
								  hyper_delta=self.hyper_delta,
								  transformation=self.transformation,
								  parametrization=self.parametrization)

		elif self.D in [3,5,6]:
			self.Model = ModelND(dimension=self.D,mu_data=self.mu_data,tau_data=self.tau_data,
								  prior=self.prior,
								  parameters=self.parameters,
								  hyper_alpha=self.hyper_alpha,
								  hyper_beta=self.hyper_beta,
								  hyper_gamma=self.hyper_gamma,
								  hyper_delta=self.hyper_delta,
								  transformation=self.transformation)
		else:
			sys.exit("Dimension not valid!")




		
	def run(self,sample_iters,burning_iters,
		init=None,
		chains=None,cores=None,
		step=None,nuts_kwargs=None,
		*args,**kwargs):
		"""
		Performs the MCMC run.
		Arguments:
		sample_iters (integer):    Number of MCMC iterations.
		burning_iters (integer):    Number of burning iterations.
		"""

		print("Computing posterior")

		with self.Model as model:
			db = pm.backends.Text(self.dir_out)

			if self.prior is "GMM" and self.parametrization == "non-central":
					step = pm.ElemwiseCategorical(vars=[model.component], values=[0, 1])
					trace = pm.sample(draws=sample_iters, 
									tune=burning_iters, 
									trace=db,
									chains=chains, cores=cores,
									discard_tuned_samples=True,
									step=[step])
			else:
				trace = pm.sample(draws=sample_iters, 
							tune=burning_iters,
							init=init,
							trace=db,
							nuts_kwargs=nuts_kwargs,
							chains=chains, cores=cores,
							discard_tuned_samples=True,
							*args,**kwargs)


	def load_trace(self,sample_iters):
		'''
		Loads a previously saved sampling of the model
		'''
		print("Loading existing chains ... ")
		with self.Model as model:
			#---------Load Trace --------------------
			all_trace = pm.backends.text.load(self.dir_out)

			#-------- Discard burn -----
			self.trace = all_trace[-sample_iters:]

		#------- Variable names -----------------------------------------------------------
		source_names = list(filter(lambda x: "source" in x, self.trace.varnames.copy()))
		global_names = list(filter(lambda x: ( ("loc" in x) 
											or ("scl" in x) 
											or ("weights" in x)
											or ("corr" in x)
											or ("beta" in x)
											or ("gamma" in x)
											or ("rt" in x)
														  ),self.trace.varnames.copy()))
		
		names = global_names.copy()

		for i,var in enumerate(names):
			test = (("interval" in var) 
					or ("log" in var)
					or ("stickbreaking" in var)
					or ("lowerbound") in var)
			if test:
				global_names.remove(var)


		names = source_names.copy()

		for i,var in enumerate(names):
			test = (("interval" in var) 
					or ("log" in var))
			if test:
				source_names.remove(var)

		self.global_names = global_names
		self.source_names = source_names
		#-------------------------------------------------------------------------------------

	def convergence(self):
		"""
		Analyse the chains.		
		"""
		print("Computing convergence statistics ...")
		dict_rhat  = pm.diagnostics.gelman_rubin(self.trace)
		dict_effn  = pm.diagnostics.effective_n(self.trace)

		print("Gelman-Rubin statistics:")
		for key,value in dict_rhat.items():
			print("{0} : {1:2.4f}".format(key,np.mean(value)))
		print("Effective sample size:")
		for key,value in dict_effn.items():
			print("{0} : {1:2.4f}".format(key,np.mean(value)))

	def plot_chains(self,
		IDs=None,
		divergences='bottom', 
		figsize=None, 
		textsize=None, 
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
		#----- Open PDFPages -------------
		pdf = PdfPages(filename=self.dir_out +"/Traces.pdf")

		if IDs is not None:
			#--------- Loop over ID in list ---------------
			for i,ID in enumerate(IDs):
				id_in_IDs = np.isin(self.ID,ID)
				if not np.any(id_in_IDs) :
					sys.exit("{0} {1} is not valid".format(self.id_name,ID))
				idx = np.where(id_in_IDs)[0]
				coords = {"1D_source_dim_0" : idx}
				plt.figure(0)
				axes = pm.plots.traceplot(self.trace,var_names=self.source_names,
						coords=coords,
						figsize=figsize,
						textsize=textsize, 
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
				pdf.savefig(bbox_inches='tight')
				plt.close(0)


		if len(self.global_names) > 0:
			plt.figure(1)
			axes = pm.plots.traceplot(self.trace,var_names=self.global_names,
					figsize=figsize,
					textsize=textsize, 
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

	def save_statistics(self,statistic,quantiles=[0.05,0.95]):
		'''
		Saves the statistics to a csv file.
		Arguments:
		dir_csv (string) Directory where to save the statistics
		statistic (string) Type of statistic (mean,median or mode)
		'''
		print("Saving statistics ...")

		#----------------------- Functions ----------------------------------

		def my_mode(sample):
			mins,maxs = np.min(sample),np.max(sample)
			x         = np.linspace(mins,maxs,num=1000)
			gkde      = st.gaussian_kde(sample.flatten())
			ctr       = x[np.argmax(gkde(x))]
			return ctr


		def trace_mean(x):
			return pn.Series(np.mean(x, 0), name='mean')

		def trace_median(x):
			return pn.Series(np.median(x, 0), name='median')

		def trace_mode(x):
			return pn.Series(np.apply_along_axis(my_mode,0,x), name='mode')

		def trace_quantiles(x):
			return pn.DataFrame(np.quantile(x,quantiles,axis=0).T,
							columns=["lower","upper"])
		#---------------------------------------------------------------------

		if statistic is "mean":
			stat_funcs = [trace_mean, trace_quantiles]
		elif statistic is "median":
			stat_funcs = [trace_median, trace_quantiles]
		elif statistic is "mode":
			stat_funcs = [trace_mode, trace_quantiles]
		else:
			sys.exit("Incorrect statistic:"+statistic)


		#-------------- Source statistics ----------------------------------------------------
		source_csv = self.dir_out +"/Sources_"+statistic+".csv"
		df_source  = pm.stats.summary(self.trace,varnames=self.source_names,stat_funcs=stat_funcs)

		#------------- Replace parameter id by source ID--------------------
		# If D is five we still infer six parameters
		D = self.D
		if self.D is 5 :
			D = 6

		n_sources = len(self.ID)
		ID  = np.repeat(self.ID,D,axis=0)
		idx = np.tile(np.arange(D),n_sources).astype('str')

		df_source.set_index(ID,inplace=True)
		df_source.insert(loc=0,column="parameter",value=idx)
		#---------------------------------------------------------------------

		#---------- Save source data frame ----------------------
		df_source.to_csv(path_or_buf=source_csv,index_label=self.id_name)

		#-------------- Global statistics ------------------------
		if len(self.global_names) > 0:
			global_csv = self.dir_out +"/Cluster_"+statistic+".csv"
			df_global = pm.stats.summary(self.trace,varnames=self.global_names,stat_funcs=stat_funcs)
			df_global.to_csv(path_or_buf=global_csv,index_label="Parameter")

	def save_samples(self):
		'''
		Saves the chain samples to an h5 file.
		Arguments:
		dir_csv (string) Directory where to save the samples
		'''
		print("Saving samples ...")

		#------ Open h5 file -------------------
		file_h5 = self.dir_out + "/Samples.h5"

		source_trace = self.trace[self.source_names[0]].T
		with h5py.File(file_h5,'w') as hf:
			grp_glb = hf.create_group("Cluster")
			grp_src = hf.create_group("Sources")

			#------ Loop over global parameters ---
			for name in self.global_names:
				label = name.replace(self.Model.name+"_","")
				grp_glb.create_dataset(label, data=self.trace[name])

			#------ Loop over source parameters ---
			for i,name in enumerate(self.ID):
				grp_src.create_dataset(name, data=source_trace[i])


	def evidence(self,N_samples=None,M_samples=1000,dlogz=1.0,nlive=None,
		quantiles=[0.05,0.95],
		print_progress=False,
		plot=False):

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

		


