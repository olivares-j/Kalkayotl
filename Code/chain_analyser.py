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
import h5py


class Analysis:
	"""
	This class provides flexibility to analyse the chains infered by emcee
	"""
	def __init__(self,file_name,names=None):
		"""
		Arguments:
		file_name (string):   Path to file containing the chains.
		chain_name (string):  Name of mcmc chain to analyse. 
					   		  Default None will loop over all names present in file_name.
		"""
		self.file_name  = file_name
		with h5py.File(file_name, 'r') as hf:
			if names is not None :
				if names not in list(hf.keys()):
					RuntimeError("{0} not in {1}".format(names,file_name))
					sys.exit(0)
			else:
				names = list(hf.keys())

		self.names = names

		#--------- Verify convergence ----------------------------
		conv = self.convergence()
		if np.logical_not(conv).any:
			RuntimeWarning("File {0} contains unconverged chains".format(self.file_name))
		#---------------------------------------------------------
		print(conv)


	def convergence(self,tol_convergence=100):
		"""
		Analyse the chains.		
		"""
		converged = np.full(len(self.names),False)
		for i,name in enumerate(self.names):
			reader = emcee.backends.HDFBackend(self.file_name,name=name)

			tau = reader.get_autocorr_time(tol=0)
			converged[i] = np.all(tau * tol_convergence < reader.iteration)

		return converged

	def get_chains(self):
			burnin = int(2*np.max(tau))
			thin = int(0.5*np.min(tau))
			samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
			log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
			log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)
			print("Not yet implemented")
			pass
		######################### TO BE INCLUDED ###########################


