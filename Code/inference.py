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
    along with PyAspidistra.  If not, see <http://www.gnu.org/licenses/>.
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
	def __init__(self,posterior,prior,prior_loc,prior_scale,**kwargs):

		self.Posterior  = posterior(prior=prior,
									prior_loc=prior_loc,
									prior_scale=prior_scale,
									**kwargs)

		self.frozen_uniform = st.uniform(loc=prior_loc,scale=prior_scale)

	def load_data(self,file_data,list_observables,*args,**kwargs):
		"""
		This function reads the data.

		Arguments:
		file_data (string): The path to a CSV file.

		list_observables (array string): Names of columns.

		Other arguments are passed to pandas.read_csv function

		"""
		#------- reads the data ----------------------------------------------
		data  = pn.read_csv(file_data,usecols=list_observables,*args,**kwargs) 
		#---------- drop na values and reorder ------------
		data  = data.dropna(thresh=2)
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
		
	def run(self,n_iter,n_walkers,file_chains="chains.h5",tol_convergence=100,progress=True):
		"""
		Performs the MCMC run.
		Arguments:
		n_iter (integer):          Number of MCMC iterations.
		n_walkers (integer):       Number of ensamble particles.
		file_chains (string):      Path to storage file.
		tol_convergence (integer): Number of autocorr times needed to ensure convergence
		progress (boolean):        Show or not a progress bar
		"""
		print("Computing posterior")
		pos0 = self.frozen_uniform.rvs(size=(n_walkers,self.Posterior.ndim))
		pbar = tqdm.tqdm(total=self.n_stars,unit="stars",disable= not progress)
		for ID,datum in self.Data.iterrows():
			self.Posterior.setup(datum)
			backend = emcee.backends.HDFBackend(file_chains,name=ID)
			sampler = emcee.EnsembleSampler(n_walkers,self.Posterior.ndim, 
						self.Posterior,
						backend=backend)
			#==========================================================================
			#-- The following follows the example provided in emcee 3.0 documentation
			#-- See https://emcee.readthedocs.io/en/latest/tutorials/monitor/
			#-- Copyright 2010-2017 Dan Foreman-Mackey and contributors.
			old_tau = np.inf
			for sample in sampler.sample(pos0,iterations=n_iter,store=True,progress=False):
				# Only check convergence every 100 steps
				if sampler.iteration % 100:
					continue

				# Compute the autocorrelation time so far
				# Using tol=0 means that we'll always get an estimate even
				# if it isn't trustworthy
				tau = sampler.get_autocorr_time(tol=0)

				# Check convergence
				converged = np.all(tau * tol_convergence < sampler.iteration)
				converged &= np.all(np.abs(old_tau - tau) / tau < (1.0/tol_convergence))
				if converged:
					break
				old_tau = tau
			#=======================================================================


			if not converged:
				print("Object: {0} did not converged!".format(ID))

			pbar.update(1)

		pbar.close()

	def analyse(self,name=None):
		"""
		Analyse the chains.

		Arguments:
		name (string): Name of mcmc chain to analyse. 
					   Default None on which all ID names are analysed.
		"""
		reader = emcee.backends.HDFBackend(filename)

		tau = reader.get_autocorr_time()
		burnin = int(2*np.max(tau))
		thin = int(0.5*np.min(tau))
		samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
		log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
		log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)
		print("Not yet implemented")
		pass
		######################### TO BE INCLUDED ###########################


