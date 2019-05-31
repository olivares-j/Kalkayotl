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
from Single import Single1D#,Single2D,Single3D,Single5D

class Inference:
	"""
	This class provides flexibility to infer the distance distribution given the parallax and its uncertainty
	"""
	def __init__(self,dimension,prior,zero_point,**kwargs):
		"""
		Arguments:
		dimension (integer):     Dimension of the inference
		prior (string):       Prior family
		prior_params (array): Prior parameters
		"""
		gaia_observables = ["ra","dec","parallax","pmra","pmdec","radial_velocity",
                    "ra_error","dec_error","parallax_error","pmra_error","pmdec_error","radial_velocity_error",
                    "ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr",
                	"dec_parallax_corr","dec_pmra_corr","dec_pmdec_corr",
                	"parallax_pmra_corr","parallax_pmdec_corr",
                	"pmra_pmdec_corr"]

		self.D          = dimension 
		self.prior      = prior
		self.zero_point = zero_point

		if self.D == 1:
			index_obs  = [2,8]
			index_mu   = [2]
			index_sd   = [8]
			index_corr = []
			

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


		elif self.D == 6:
			index_obs  = range(22)
			index_mu   = [0,1,2,3,4,5]
			index_sd   = [6,7,8,9,10,11]
			index_corr = [12,13,14,15,16,17,18,19,20,21]

		else:
			sys.exit("Dimension not valid!")

		self.names_obs  = [gaia_observables[i] for i in index_obs]
		self.names_mu   = [gaia_observables[i] for i in index_mu]
		self.names_sd   = [gaia_observables[i] for i in index_sd]
		self.names_corr = [gaia_observables[i] for i in index_corr] 


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

		#================== Correct units ==========================
		#----- mas to sec -----------------------------------
		data["parallax"]       = data["parallax"]*1e-3
		data["parallax_error"] = data["parallax_error"]*1e-3

		if self.D > 1:
			#----- Transform mas to deg ------------------
			data["ra_error"]  = data["ra_error"]/3.6e6
			data["dec_error"] = data["dec_error"]/3.6e6 
		#============================================================

		#----- put ID as row name-----
		data.set_index(list_observables[0],inplace=True)

		self.n_stars,D = np.shape(data)
		if D != 2 :
			RuntimeError("Data have incorrect shape!")

		#==================== Set Mu and Sigma =========================================
		Mu    = np.zeros(self.n_stars*self.D)
		Sigma = np.zeros((self.n_stars*self.D,self.n_stars*self.D))
		for i,(ID,datum) in enumerate(data.iterrows()):
			# idx = [b for b in range(i*self.D,i*self.D + self.D)]
			idx = range(i*self.D,i*self.D + self.D)
			# ra,dec,pax,pmra,pmdec,rvel                            = datum[:6]
			# u_ra,u_dec,u_pax,u_pmra,u_pmdec,u_rvel                = datum[6:12]
			# ra_dec_corr,ra_pax_corr,ra_pmra_corr,ra_pmdec_corr    = datum[12:16]
			# dec_pax_corr,dec_pmra_corr,dec_pmdec_corr             = datum[16:19]
			# pax_pmra_corr,pax_pmdec_corr                          = datum[19:21]
			# pmra_pmdec_corr                                       = datum[21]
			mu   = np.array(datum[self.names_mu])  - self.zero_point
			sd   = np.array(datum[self.names_sd])
			corr = np.array(datum[self.names_corr])

			#-------- Correlation matrix ---------
			rho  = np.zeros((self.D,self.D))
			rho  = rho + rho.T + np.eye(self.D)

			#-------- Covariance matrix ----------------------
			sigma = np.diag(sd).dot(rho.dot(np.diag(sd)))
			
			#---------- Include mu and sigma in Mu and Sigma ---
			Mu[idx] = mu
			Sigma[np.ix_(idx,idx)] = sigma

		self.mu_data    = Mu
		self.sigma_data = Sigma
		#=================================================================================

		print("Data correctly loaded")

	def setup(self):
		'''
		Set-up the model with the corresponding dimensions and data
		'''

		if self.D == 1:
			self.Model = Single1D(mu_data=self.mu_data,Sigma_data=self.sigma_data,
								  prior=self.prior)
			
		elif self.D == 3:
			self.Model = Single3D(mu_data=self.mu_data,Sigma_data=self.sigma_data,
								  prior=self.prior)
			
		elif self.D == 5:
			self.Model = Single5D(mu_data=self.mu_data,Sigma_data=self.sigma_data,
								  prior=self.prior)
	
		elif self.D == 6:
			self.Model = Single6D(mu_data=self.mu_data,Sigma_data=self.sigma_data,
								  prior=self.prior)
		else:
			sys.exit("Dimension not valid!")


		
	def run(self,sample_iters,burning_iters,file_chains="chains.h5"):
		"""
		Performs the MCMC run.
		Arguments:
		sample_iters (integer):    Number of MCMC iterations.
		burning_iters (integer):    Number of burning iterations.
		file_chains (string):      Path to storage file.
		"""
		print("Computing posterior")
		with self.Model as model:
			trace = pm.sample(sample_iters, tune=burning_iters)
		


