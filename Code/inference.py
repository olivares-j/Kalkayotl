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
from Models import Model1D,Model3D,Model5D,Model6D

from Functions import AngularSeparation,CovarianceParallax,CovariancePM

class Inference:
	"""
	This class provides flexibility to infer the distance distribution given the parallax and its uncertainty
	"""
	def __init__(self,dimension,prior,parameters,
				hyper_alpha,
				hyper_beta,
				hyper_gamma,
				hyper_delta,
				transformation,
				zero_point,**kwargs):
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
		self.transformation   = transformation

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


	def load_data(self,file_data,id_name,*args,**kwargs):
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
			data["ra_error"]       = data["ra_error"]/(1e3*3600.0)  + 10.0/3600.0
			data["dec_error"]      = data["dec_error"]/(1e3*3600.0) + 10.0/3600.0
		#============================================================

		#----- put ID as row name-----
		data.set_index(list_observables[0],inplace=True)

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


		#===================== Set correlations amongst stars ===========================
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
			
		elif self.D == 3:
			self.Model = Model3D(mu_data=self.mu_data,sg_data=self.sg_data,
								  prior=self.prior,
								  parameters=self.parameters,
								  hyper_alpha=self.hyper_alpha,
								  hyper_beta=self.hyper_beta,
								  hyper_gamma=self.hyper_gamma,
								  hyper_delta=self.hyper_delta,
								  transformation=self.transformation)
			
		elif self.D == 5:
			self.Model = Model5D(mu_data=self.mu_data,sg_data=self.sg_data,
								  prior=self.prior,
								  parameters=self.parameters,
								  hyper_alpha=self.hyper_alpha,
								  hyper_beta=self.hyper_beta,
								  hyper_gamma=self.hyper_gamma,
								  hyper_delta=self.hyper_delta,
								  transformation=self.transformation)
	
		elif self.D == 6:
			self.Model = Model6D(mu_data=self.mu_data,sg_data=self.sg_data,
								  prior=self.prior,
								  parameters=self.parameters,
								  hyper_alpha=self.hyper_alpha,
								  hyper_beta=self.hyper_beta,
								  hyper_gamma=self.hyper_gamma,
								  hyper_delta=self.hyper_delta,
								  transformation=self.transformation)
		else:
			sys.exit("Dimension not valid!")


		
	def run(self,sample_iters,burning_iters,dir_chains):
		"""
		Performs the MCMC run.
		Arguments:
		sample_iters (integer):    Number of MCMC iterations.
		burning_iters (integer):    Number of burning iterations.
		file_chains (string):      Path to storage file.
		"""
		print("Computing posterior")
		with self.Model as model:
			db = pm.backends.Text(dir_chains)
			trace = pm.sample(sample_iters, tune=burning_iters, trace=db)
		


