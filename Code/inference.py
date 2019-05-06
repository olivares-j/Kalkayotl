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
import emcee
import numpy as np
import pandas as pn
import scipy.stats as st
import tqdm

class Inference:
	"""
	This class provides flexibility to infer the distance distribution given the parallax and its uncertainty
	"""
	def __init__(self,posterior,prior,prior_loc,prior_scale,n_walkers,**kwargs):
		"""
		Arguments:
		n_walkers (integer):       Number of ensamble particles.
		"""
		gaia_observables = ["ra","dec","parallax","pmra","pmdec","radial_velocity",
                    "ra_error","dec_error","parallax_error","pmra_error","pmdec_error","radial_velocity_error",
                    "ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr",
                	"dec_parallax_corr","dec_pmra_corr","dec_pmdec_corr",
                	"parallax_pmra_corr","parallax_pmdec_corr",
                	"pmra_pmdec_corr"]

		self.Posterior  = posterior(prior=prior,
									prior_loc=prior_loc,
									prior_scale=prior_scale,
									**kwargs)
		self.n_walkers = n_walkers

		if self.Posterior.ndim == 1:
			index_observables = [2,8]
			self.pos0 = st.uniform.rvs(loc=prior_loc,scale=prior_scale,size=(n_walkers,self.Posterior.ndim))

		if self.Posterior.ndim == 3:
			index_observables = [0,1,2,6,7,8,12,13,16]
			self.pos0 = [np.array([st.uniform.rvs(loc=0,scale=360,size=1)[0],
								   st.uniform.rvs(loc=-90,scale=180,size=1)[0],
								   st.uniform.rvs(loc=prior_loc,scale=prior_scale,size=1)[0]]) for i in range(n_walkers)]

		if self.Posterior.ndim == 5:
			index_observables = [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21]
			self.pos0 = [np.array([st.uniform.rvs(loc=0,scale=360,size=1)[0],
								   st.uniform.rvs(loc=-90,scale=180,size=1)[0],
								   st.uniform.rvs(loc=prior_loc,scale=prior_scale,size=1)[0],
								   st.cauchy.rvs(loc=0.0,scale=500.0,size=1)[0],
								   st.cauchy.rvs(loc=0.0,scale=500.0,size=1)[0]]) for i in range(n_walkers)]

		if self.Posterior.ndim == 6:
			index_observables = range(22)
			self.pos0 = [np.array([st.uniform.rvs(loc=0,scale=360,size=1)[0],
								   st.uniform.rvs(loc=-90,scale=180,size=1)[0],
								   st.uniform.rvs(loc=prior_loc,scale=prior_scale,size=1)[0],
								   st.cauchy.rvs(loc=0.0,scale=500.0,size=1)[0],
								   st.cauchy.rvs(loc=0.0,scale=500.0,size=1)[0],
								   st.cauchy.rvs(loc=0.0,scale=300.0,size=1)[0]]) for i in range(n_walkers)]

		self.observables = [gaia_observables[i] for i in index_observables]


	def load_data(self,file_data,id_name,*args,**kwargs):
		"""
		This function reads the data.

		Arguments:
		file_data (string): The path to a CSV file.

		id_name (string): Identifier in file_csv.

		Other arguments are passed to pandas.read_csv function

		"""
		list_observables = sum([[id_name],self.observables],[])

		#------- reads the data ----------------------------------------------
		data  = pn.read_csv(file_data,usecols=list_observables,*args,**kwargs) 
		#---------- drop na values and reorder ------------
		data  = data.dropna(thresh=len(list_observables))
		data  = data.reindex(columns=list_observables)

		#------- index as string ------
		data[list_observables[0]] = data[list_observables[0]].astype('str')

		#------- Correct units ------
		data["parallax"]       = data["parallax"]*1e-3
		data["parallax_error"] = data["parallax_error"]*1e-3

		#----- put ID as row name-----
		data.set_index(list_observables[0],inplace=True)

		self.n_stars,D = np.shape(data)
		if D != 2 :
			RuntimeError("Data have incorrect shape!")

		self.Data = data
		print("Data correctly loaded")
		
	def run(self,n_iter,file_chains="chains.h5",tol_convergence=10,progress=True):
		"""
		Performs the MCMC run.
		Arguments:
		n_iter (integer):          Number of MCMC iterations.
		file_chains (string):      Path to storage file.
		tol_convergence (integer): Number of autocorr times needed to ensure convergence
		progress (boolean):        Show or not a progress bar
		"""
		print("Computing posterior")
		pbar = tqdm.tqdm(total=self.n_stars,unit="stars",disable= not progress)
		for ID,datum in self.Data.iterrows():
			self.Posterior.setup(datum)
			backend = emcee.backends.HDFBackend(file_chains,name=ID)
			sampler = emcee.EnsembleSampler(self.n_walkers,self.Posterior.ndim, 
						self.Posterior,
						backend=backend)
			#==========================================================================
			#-- The following follows the example provided in emcee 3.0 documentation
			#-- See https://emcee.readthedocs.io/en/latest/tutorials/monitor/
			#-- Copyright 2010-2017 Dan Foreman-Mackey and contributors.
			old_tau = np.inf
			for sample in sampler.sample(self.pos0,iterations=n_iter,store=True,progress=False):
				# Only check convergence every 100 steps
				if sampler.iteration % 100:
					continue

				# Compute the autocorrelation time so far
				# Using tol=0 means that we'll always get an estimate even
				# if it isn't trustworthy
				tau = sampler.get_autocorr_time(tol=0)

				# Check convergence
				converged  = np.all(tau * tol_convergence < sampler.iteration)
				converged &= np.all(np.abs(1.0 - (tau/old_tau)) < (1.0/tol_convergence))
				if converged:
					break
				old_tau = tau
			#=======================================================================


			if not converged:
				print("Object: {0} did not converged!".format(ID))

			pbar.update(1)

		pbar.close()


