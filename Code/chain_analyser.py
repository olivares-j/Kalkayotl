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
import corner 

class Analysis:
	"""
	This class provides flexibility to analyse the chains infered by emcee
	"""
	def __init__(self,file_name,names=None,id_name="ID",dir_plots="Plots/",
		figsize=(6.3,6.3),
		quantiles=[0.159,0.5,0.841]):
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
		self.figure_size = figsize

		if not os.path.isdir(dir_plots):
			os.mkdir(dir_plots)

		self.dir_plots  = dir_plots
		self.id         = str(id_name)

		self.quantiles  = quantiles

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		
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
		labels_1d = ["Dist. [pc]"]
		labels_3d = ["R.A. [deg]","Dec. [deg]","Dist. [pc]"]
		#---------- Loop over names in file ---------------------------
		for i,name in enumerate(self.names):
			print("Plotting object: {0}".format(name))
			reader = emcee.backends.HDFBackend(self.file_name,name=name)
			tau    = reader.get_autocorr_time(tol=0)
			burnin = int(burnin_tau*np.max(tau))
			sample = reader.get_chain(discard=burnin)

			#-------------- Properties of chain -----------------
			N,walkers,D = sample.shape
			if D == 1 :
				labels = labels_1d
			elif D == 3:
				labels = labels_3d
			else:
				print("Error")
			#----------------------------------------------------------

			pdf = PdfPages(filename=self.dir_plots+self.id+"_"+str(name)+".pdf")
			
			#----------- Trace plots --------------------------
			if D == 1:
				plt.plot(sample[:, :, 0], "k", alpha=0.3)
				plt.xlim(0,N)
				plt.ylabel(labels_1d[0])
				plt.xlabel("Step")

			else :
				fig, axes = plt.subplots(D, figsize=self.figure_size, sharex=True)
				for i in range(D):
				    ax = axes[i]
				    ax.plot(sample[:, :, i], "k", alpha=0.3)
				    ax.set_xlim(0,N)
				    ax.set_ylabel(labels[i])
				    ax.yaxis.set_label_coords(-0.1, 0.5)

				axes[-1].set_xlabel("Step")

			#-------------- Save fig --------------------------
			pdf.savefig(bbox_inches='tight')
			plt.close()

			#----------- Corner plot --------------------------
			sample = sample.reshape((N*walkers,D))
			fig = corner.corner(sample, 
						labels=labels,
						show_titles=True,
						use_math_text=True,
						quantiles=self.quantiles)
			fig.set_size_inches(self.figure_size)
			#----------------------------------------------
			pdf.savefig(bbox_inches='tight')
			plt.close()
			pdf.close()

	def get_statistics(self,burnin_tau=3.0):
		"""
		This function computes the chain statistics.

		Arguments:
		burnin_tau (float): Number of autocorr times to discard as burnin.

		Returns (array):    For each parameter it contains min,central, and max. 
		"""
		stats = np.zeros((len(self.names),D,3))
		#---------- Loop over names in file ---------------------------
		for i,name in enumerate(self.names):
			reader = emcee.backends.HDFBackend(self.file_name,name=name)
			tau    = reader.get_autocorr_time(tol=0)
			burnin = int(burnin_tau*np.max(tau))
			sample = reader.get_chain(discard=burnin)
			logpro = reader.get_log_prob(discard=burnin)

			N,walkers,D = sample.shape

			#--------- MAP -------------
			idx_map = np.unravel_index(logpro.argmax(), logpro.shape)
			MAP     = sample[idx_map]
			#-------------------------
			
			for d in range(D):
				stats[i,d] = corner.quantile(sample[:,:,i],self.quantiles)
		
		return stats


			


