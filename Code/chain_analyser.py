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
import sys
import os
import warnings

import emcee
import numpy as np
import h5py

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter

from chainconsumer import ChainConsumer

class Analysis:
	"""
	This class provides flexibility to analyse the chains infered by emcee
	"""
	def __init__(self,file_name,names=None,statistics="max",id_name="ID",dir_plots="Plots/"):
		"""
		Arguments:
		file_name (string):   Path to file containing the chains.
		labels (list of strings): Labels, starting from identifier and 
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
		if np.logical_not(conv).any():
			warnings.warn("File {0} contains unconverged chains".format(self.file_name),Warning)
		#---------------------------------------------------------
		
		#---------- Plots -------------
		self.figure_size = (6.3, 6.3)

		if not os.path.isdir(dir_plots):
			os.mkdir(dir_plots)

		self.dir_plots  = dir_plots
		self.id         = str(id_name)

		#------- ChainConsumer ------------------
		self.cc         = ChainConsumer()
		self.statistics = statistics
		
	def convergence(self,tol_convergence=100):
		"""
		Analyse the chains.		
		"""
		converged = np.full(len(self.names),False)
		for i,name in enumerate(self.names):
			reader = emcee.backends.HDFBackend(self.file_name,name=name)

			tau = reader.get_autocorr_time(tol=0)
			converged[i] = np.all((tau * tol_convergence) < reader.iteration)

		return converged

	def plot_chains(self,burnin_tau=2.0):
		"""
		This function plots the chains and the marginals

		Arguments:
		burnin_tau (float): Number of autocorr times to discard as burnin.
		"""
		#---------- Loop over names in file ---------------------------
		for i,name in enumerate(self.names):
			reader = emcee.backends.HDFBackend(self.file_name,name=name)
			tau    = reader.get_autocorr_time(tol=0)
			burnin = int(burnin_tau*np.max(tau))
			sample = reader.get_chain(discard=burnin, flat=True)

			#----------- Adds chain -------------------------------
			self.cc.add_chain(sample, parameters=["Distance [pc]"])
			self.cc.configure(statistics=self.statistics,
								colors=["black"],
								marker_style="-",
								marker_alpha=1.0,
								linewidths=0.5,
								shade_alpha=1.0)
			#------------------------------------------------------------------
			pdf = PdfPages(filename=self.dir_plots+self.id+"_"+str(name)+".pdf")
			extents = [[0.95*np.min(sample),1.05*np.max(sample)]]
			#----------- Trace plots --------------------------
			fig = self.cc.plotter.plot_walks(extents=extents)
			pdf.savefig(bbox_inches='tight')
			plt.close()
			#----------- Corner plot --------------------------
			fig = self.cc.plotter.plot(figsize="PAGE")
			pdf.savefig(bbox_inches='tight')
			plt.close()
			pdf.close()
			self.cc.remove_chain()
			


