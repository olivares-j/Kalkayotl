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
import pymc3 as pm
import numpy as np
import pandas as pn
from Models import Model1D,ModelND

from Functions import AngularSeparation,CovarianceParallax,CovariancePM


import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random

import scipy.stats as st


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


	def load_data(self,file_data,id_name,radec_inflation=10.0,id_length=10,*args,**kwargs):
		"""
		This function reads the data.

		Arguments:
		file_data (string): The path to a CSV file.

		id_name (string): Identifier in file_csv.

		Other arguments are passed to pandas.read_csv function

		"""
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
		# In case of missing values
		IDs = []
		#----------------------------

		self.n_stars,D = np.shape(data)
		if D != 2 :
			RuntimeError("Data have incorrect shape!")

		#==================== Set Mu and Sigma =========================================
		self.mu_data = np.zeros(self.n_stars*self.D)
		self.sg_data = np.zeros((self.n_stars*self.D,self.n_stars*self.D))
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
			self.mu_data[ida] = mu
			self.sg_data[np.ix_(ida,ida)] = sigma
		#=========================================================================

		#----- Save identifiers ------
		df_IDs = pn.DataFrame(IDs,columns=["ID"])
		df_IDs.to_csv(path_or_buf=self.dir_out+"/Identifiers.csv",index_label="Parameter")
		self.ID = IDs



		#===================== Set correlations amongst stars ===========================
		if not self.indep_measures :
			#------ Obtain array of positions ------------
			positions = data[["ra","dec"]].to_numpy()

			#------ Angular separations ----------
			theta = AngularSeparation(positions)

			#------ Covariance in parallax -----
			cov_plx = CovarianceParallax(theta)
			np.fill_diagonal(cov_plx,0.0)

			#------ Add parallax covariance -----------------------
			ida_plx = [i*self.D + self.idx_plx for i in range(self.n_stars)]
			self.sg_data[np.ix_(ida_plx,ida_plx)] += cov_plx
			#------------------------------------------------------
			
			if self.D > 3:
				#------ Covariance in PM ----------------------------
				# Same for mu_alpha and mu_delta
				cov_pms = CovariancePM(theta)
				np.fill_diagonal(cov_pms,0.0)

				#------ Add PM covariances -----------------------
				ida_pma = [i*self.D + self.idx_pma for i in range(self.n_stars)]
				ida_pmd = [i*self.D + self.idx_pmd for i in range(self.n_stars)]

				self.sg_data[np.ix_(ida_pma,ida_pma)] += cov_pms
				self.sg_data[np.ix_(ida_pmd,ida_pmd)] += cov_pms


		#=================================================================================
		print("Data correctly loaded")


	def setup(self):
		'''
		Set-up the model with the corresponding dimensions and data
		'''

		if self.D == 1:
			self.Model = Model1D(mu_data=self.mu_data,sg_data=self.sg_data,
								  prior=self.prior,
								  parameters=self.parameters,
								  hyper_alpha=self.hyper_alpha,
								  hyper_beta=self.hyper_beta,
								  hyper_gamma=self.hyper_gamma,
								  hyper_delta=self.hyper_delta,
								  transformation=self.transformation)

		elif self.D in [3,5,6]:
			self.Model = ModelND(dimension=self.D,mu_data=self.mu_data,sg_data=self.sg_data,
								  prior=self.prior,
								  parameters=self.parameters,
								  hyper_alpha=self.hyper_alpha,
								  hyper_beta=self.hyper_beta,
								  hyper_gamma=self.hyper_gamma,
								  hyper_delta=self.hyper_delta,
								  transformation=self.transformation)
		else:
			sys.exit("Dimension not valid!")


		
	def run(self,sample_iters,burning_iters):
		"""
		Performs the MCMC run.
		Arguments:
		sample_iters (integer):    Number of MCMC iterations.
		burning_iters (integer):    Number of burning iterations.
		file_chains (string):      Path to storage file.
		"""
		print("Computing posterior")
		with self.Model as model:
			db = pm.backends.Text(self.dir_out)
			trace = pm.sample(sample_iters, tune=burning_iters, trace=db,
							  discard_tuned_samples=True)

	def load_trace(self,burning_iters):
		'''
		Loads a previously saved sampling of the model
		'''
		print("Loading existing chains ... ")
		with self.Model as model:
			#---------Load Trace --------------------
			all_trace = pm.backends.text.load(self.dir_out)

			#-------- Discard burn -----
			self.trace = all_trace[burning_iters:]

		print("The following chains have been loaded:")
		print(self.trace)

		#------- Variable names -----------------------------------------------------------
		source_names = list(filter(lambda x: "source" in x, self.trace.varnames.copy()))
		global_names = list(filter(lambda x: ( ("loc" in x) 
											or ("scl" in x) 
											or ("weights" in x)
											or ("corr" in x)
														  ),self.trace.varnames.copy()))
		
		names = global_names.copy()

		for i,var in enumerate(names):
			test = (("interval" in var) 
					or ("log" in var)
					or ("stickbreaking") in var)
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

	def plot_chains(self,dir_plots,
		coords=None,
		divergences='bottom', 
		figsize=None, 
		textsize=None, 
		lines=None, 
		combined=False, 
		plot_kwargs=None, 
		hist_kwargs=None, 
		trace_kwargs=None):
		"""
		This function plots the trace. Parameters are the same as in pymc3
		"""

		print("Plotting traces ...")

		pdf = PdfPages(filename=dir_plots+"/Traces.pdf")


		plt.figure(1)
		pm.plots.traceplot(self.trace,var_names=self.source_names,
			coords=coords,
			figsize=figsize,
			textsize=textsize, 
			lines=lines, 
			combined=combined, 
			plot_kwargs=plot_kwargs, 
			hist_kwargs=hist_kwargs, 
			trace_kwargs=trace_kwargs)
			
		#-------------- Save fig --------------------------
		pdf.savefig(bbox_inches='tight')
		plt.close(1)

		if len(self.global_names) > 0:
			plt.figure(2)
			pm.plots.traceplot(self.trace,var_names=self.global_names,
				figsize=figsize,
				textsize=textsize, 
				lines=lines, 
				combined=combined, 
				plot_kwargs=plot_kwargs, 
				hist_kwargs=hist_kwargs, 
				trace_kwargs=trace_kwargs)
				
			#-------------- Save fig --------------------------
			pdf.savefig(bbox_inches='tight')
			plt.close(2)

		
		pdf.close()

	def save_statistics(self,dir_csv,statistic,quantiles=[0.159,0.841]):
		'''
		Saves the statistics to a csv file.
		Arguments:
		dir_csv (string) Directory where to save the statistics
		statistic (string) Type of statistic (mean,median or mode)
		quantiles (list of floats) Quantile values to return

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
			return pn.DataFrame(np.quantile(x, quantiles,axis=0).T,
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
		source_csv = dir_csv +"/Sources_"+statistic+".csv"
		df_source  = pm.stats.summary(self.trace,varnames=self.source_names,stat_funcs=stat_funcs)

		#------------- Replace parameter id by source ID--------------------
		# If D is five we still infer six parameters
		D = self.D
		if self.D is 5 :
			D = 6

		n_sources = len(self.ID)
		A = np.char.array(np.repeat(self.ID,D,axis=0))
		B = np.char.array(np.repeat("__",D*n_sources))
		C = np.char.array(np.tile(np.arange(D),n_sources).astype('str'))
		ID = A + B + C

		df_source.set_index(ID,inplace=True)
		#---------------------------------------------------------------------

		#---------- Save source data frame ----------------------
		df_source.to_csv(path_or_buf=source_csv,index_label="ID")

		#-------------- Global statistics ------------------------
		if len(self.global_names) > 0:
			global_csv = dir_csv +"/Cluster_"+statistic+".csv"
			df_global = pm.stats.summary(self.trace,varnames=self.global_names,stat_funcs=stat_funcs)
			df_global.to_csv(path_or_buf=global_csv,index_label="Parameter")
		


