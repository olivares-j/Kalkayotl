'''
Copyright 2018 Javier Olivares Romero

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

class Inference(object):
	"""
	This class provides flexibility to infer the distance distribution given the parallax and its uncertainty
	"""
	def __init__(self,prior,prior_loc,prior_scale,*args,**kwargs):

		self.Posterior    = super.__init__(prior,prior_loc,prior_scale,*args,**kwargs)

		self.frozen_uniform = st.uniform.rvs(loc=prior_loc,scale=prior_scale)

	def load_data(self,file_data,list_observables,*args,**kwargs):
		"""
		This function reads the data.
		string  file_data: Must be a CSV file with identifier, parallax and uncertainty
				UNITS in miliarcseconds.

		strings list_observables: Names of columns containing in order:
					IDENTIFIER, PARALLAX, and PARALLAX uncertainties

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
		
	def run(self,n_iter,n_walkers,file_chains="chains.h5",progress=True):
		"""
		Performs the MCMC run
		"""
		pos0 = self.frozen_uniform.rvs(size=(n_walkers,self.ndim))
		pbar = tqdm.tqdm(total=self.n_stars,unit="stars",disable=!progress)
		for ID,datum in data.iterrows():
			self.Posterior.setup(datum)
			backend = emcee.backends.HDFBackend(file_chains,name=ID)
			sampler = emcee.EnsembleSampler(n_walkers,self.ndim, self.Posterior,
						backend=backend,store=True,progress=False)
			sampler.run_mcmc(pos0,n_iter)
			pbar.update(1)
		pbar.close()

	def analyse(self):
		"""
		Analyse the chains
		"""
		######################### TO BE INCLUDED ###########################


