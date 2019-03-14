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
import pandas as pn
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
	def __init__(self,file_name,n_dim=1,names=None,id_name="ID",dir_plots="Plots/",
		figsize=(6.3,6.3),
		quantiles=[0.159,0.5,0.841],
		tol_convergence=100):
		"""
		Arguments:
		n_dim (int):          Number of parameter dimensions.
		file_name (string):   Path to file containing the chains.
		labels (list of strings): Labels, starting from identifier and 
		chain_name (string):  Name of mcmc chain to analyse. 
					   		  Default None will loop over all names present in file_name.
		"""
		suffix = ["_min","_ctr","_max"]

		if n_dim == 1:
			self.labels     = ["Dist. [pc]"]
			prefix          = "dist"
			self.labels_csv = [ prefix+su for su in suffix]
		elif n_dim == 3:
			self.labels     = ["R.A. [deg]","Dec. [deg]","Dist. [pc]"]
			prefix          = ["ra","dec","dist"]
			self.labels_csv = sum([[pre+su for su in suffix] for pre in prefix],[])
		else:
			print("Error. Not yet implemented!")

		self.n_dim = n_dim

		self.file_name  = file_name
		with h5py.File(file_name, 'r') as hf:
			if names is not None :
				if names not in list(hf.keys()):
					RuntimeError("{0} not in {1}".format(names,file_name))
					sys.exit(0)
			else:
				names = list(hf.keys())

		self.names = sorted(names,key=lambda x:int(x))
		self.quantiles  = quantiles
		
		#---------- Plots -------------
		self.figure_size = figsize

		if not os.path.isdir(dir_plots):
			os.mkdir(dir_plots)

		self.dir_plots  = dir_plots
		self.id         = str(id_name)

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		#------------------------------------------

		#--------- Verify convergence ----------------------------
		conv = self.convergence(tol_convergence)
		if np.logical_not(conv).any():
			warnings.warn("File {0} contains unconverged chains".format(self.file_name),Warning)
		#---------------------------------------------------------
		
	def convergence(self,tol_convergence):
		"""
		Analyse the chains.		
		"""
		converged = np.full(len(self.names),False)
		for i,name in enumerate(self.names):
			reader = emcee.backends.HDFBackend(self.file_name,name=name)

			tau = reader.get_autocorr_time(tol=0)
			converged[i] = np.all((tau * tol_convergence) < reader.iteration)

		return converged

	def plot_chains(self,names=None,burnin_tau=2.0,true_values=None,use_map=False,title_fmt=".2f"):
		"""
		This function plots the chains and the marginals

		Arguments:
		burnin_tau (float): Number of autocorr times to discard as burnin.
		true_values (float): Numpy array with true values.
		"""
		if names is None:
			names = self.names

		if true_values is not None:
			n_t,d_t = true_values.shape
			assert n_t == len(names), 'True values must be of same length as names'
			assert d_t == self.n_dim, 'True values must be of same dimension as parameters'
		#---------- Loop over names in file ---------------------------
		for i,name in enumerate(names):
			print("Plotting object: {0}".format(name))
			reader = emcee.backends.HDFBackend(self.file_name,name=name)
			tau    = reader.get_autocorr_time(tol=0)
			burnin = int(burnin_tau*np.max(tau))
			sample = reader.get_chain(discard=burnin)

			if use_map:
				#--------- MAP -------------------------------------------
				logpro  = reader.get_log_prob(discard=burnin)
				idx_map = np.unravel_index(logpro.argmax(), logpro.shape)
				MAP     = sample[idx_map]
				#---------------------------------------------------------
			

			#-------------- Properties of chain -----------------
			N,walkers,D = sample.shape

			assert self.n_dim == D, "The dimension in the chain differs from that specified!"
			
			#----------------------------------------------------------

			pdf = PdfPages(filename=self.dir_plots+self.id+"_"+str(name)+".pdf")
			plt.figure(1)
			#----------- Trace plots --------------------------
			if self.n_dim == 1:
				plt.plot(sample[:, :, 0], "k", alpha=0.3)
				plt.xlim(0,N)
				plt.ylabel(self.labels[0])
				plt.xlabel("Step")

			else :
				fig, axes = plt.subplots(self.n_dim, figsize=self.figure_size, sharex=True)
				for j in range(self.n_dim):
				    ax = axes[j]
				    ax.plot(sample[:, :, j], "k", alpha=0.3)
				    ax.set_xlim(0,N)
				    ax.set_ylabel(self.labels[j])
				    ax.yaxis.set_label_coords(-0.1, 0.5)

				axes[-1].set_xlabel("Step")

			#-------------- Save fig --------------------------
			pdf.savefig(bbox_inches='tight')
			plt.close(1)
			
			#----------- Corner plot --------------------------
			sample = sample.reshape((N*walkers,self.n_dim))
			plt.figure(1)
			fig = corner.corner(sample, 
						labels=self.labels,
						truths=true_values[i],
						use_math_text=True)

			#=========== Titles =========================================
			# Extract the axes
			axes = np.array(fig.axes).reshape((self.n_dim, self.n_dim))

			# Loop over the diagonal
			for i in range(self.n_dim):
				ax = axes[i, i]
				
				q_low, q_ctr, q_max = self.statistics(sample[:,i])
				if use_map:
					q_ctr = MAP[i]
				q_m, q_p = q_ctr-q_low, q_max-q_ctr

				# Format the quantile display.
				fmt = "{{0:{0}}}".format(title_fmt).format
				title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
				title = title.format(fmt(q_ctr), fmt(q_m), fmt(q_p))
				ax.set_title(title)
			#===========================================================

			fig.set_size_inches(self.figure_size)
			#----------------------------------------------
			pdf.savefig(bbox_inches='tight')
			plt.close("all")
			pdf.close()

	def get_statistics(self,burnin_tau=3.0,use_map=False):
		"""
		This function computes the chain statistics.

		Arguments:
		burnin_tau (float): Number of autocorr times to discard as burnin.

		Returns (array):    For each parameter it contains min,central, and max. 
		"""
		stats = np.zeros((len(self.names),self.n_dim*3))
		#---------- Loop over names in file ---------------------------
		for i,name in enumerate(self.names):
			reader = emcee.backends.HDFBackend(self.file_name,name=name)
			tau    = reader.get_autocorr_time(tol=0)
			burnin = int(burnin_tau*np.max(tau))
			sample = reader.get_chain(discard=burnin)

			N,walkers,D = sample.shape
			assert self.n_dim == D, "The dimension in the chain differs from that specified!"
			
			for d in range(self.n_dim):
				stats[i,(3*d):(3*d + 3)] = self.statistics(sample[:,:,d])

		if use_map:
			if self.n_dim == 1:
				stats[:,1]  = self.get_MAP(burnin_tau=burnin_tau)
			elif self.n_dim == 3:
				stats[:,[1,4,7]]  = self.get_MAP(burnin_tau=burnin_tau)
			else:
				print("Not yet implemented!")
		
		return stats

	def get_MAP(self,burnin_tau=3.0):
		"""
		This function computes the MAP of each parameter.

		Arguments:
		burnin_tau (float): Number of autocorr times to discard as burnin.

		Returns (array):    For each parameter it returns the MAP. 
		"""
		stats = np.zeros((len(self.names),self.n_dim))
		#---------- Loop over names in file ---------------------------
		for i,name in enumerate(self.names):
			reader = emcee.backends.HDFBackend(self.file_name,name=name)
			tau    = reader.get_autocorr_time(tol=0)
			burnin = int(burnin_tau*np.max(tau))
			sample = reader.get_chain(discard=burnin)
			logpro = reader.get_log_prob(discard=burnin)

			N,walkers,D = sample.shape
			assert self.n_dim == D, "The dimension in the chain differs from that specified!"

			#--------- MAP -------------
			idx_map  = np.unravel_index(logpro.argmax(), logpro.shape)
			stats[i] = sample[idx_map]
			#-------------------------

		if self.n_dim == 1:
			stats = stats.flatten()
		return stats


	def statistics(self,sample):
		'''
		Computes the statistics of the parameter
		'''
		return corner.quantile(sample,self.quantiles)


	def save_statistics(self,file_csv,**kwargs):
		'''
		Saves the statistics to a csv file.
		'''

		stats = self.get_statistics(**kwargs)

		df_stats = pn.DataFrame(data=stats,columns=self.labels_csv)
		df_id    = pn.DataFrame(data=self.names,columns=[self.id])

		df       = df_id.join(df_stats)
		df.to_csv(path_or_buf=file_csv,index=False)
		


			


