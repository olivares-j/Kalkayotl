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

import scipy.stats as st

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import corner
from tqdm import tqdm

from pygaia.astrometry.coordinates import Transformations
from pygaia.astrometry.coordinates import CoordinateTransformation
from pygaia.astrometry.vectorastrometry import sphericalToCartesian
from pygaia.astrometry.vectorastrometry import astrometryToPhaseSpace

def Identity3D(a,b,c):
	return a,b,c

class Analysis:
	"""
	This class provides flexibility to analyse the chains inferred by emcee
	"""
	def __init__(self,file_name,n_dim=1,names=None,id_name="ID",dir_plots="Plots/",
		figsize=(10,10),
		quantiles=[0.159,0.841],
		statistic="median",
		tol_convergence=100,
		burnin_tau=3.0,
		npts_mode=1000,
		transformation=None):
		"""
		Arguments:
		n_dim (int):          Number of parameter dimensions.
		file_name (string):   Path to file containing the chains.
		labels (list of strings): Labels, starting from identifier and 
		chain_name (string):  Name of mcmc chain to analyse. 
					   		  Default None will loop over all names present in file_name.
		transformation:       Either None or one of ["XYZ","ICRS2GAL","ICRS2ECL"]
		"""
		

		self.transformation = transformation
		self.n_dim          = n_dim
		self.npts_mode      = npts_mode
		self.file_name      = file_name
		self.quantiles      = quantiles
		self.statistic      = statistic
		self.figure_size    = figsize
		self.dir_plots      = dir_plots
		self.id             = str(id_name)
		self.burnin_tau     = burnin_tau


		#===================== Labels ====================================================

		suffix = ["_min","_ctr","_max"]

		if n_dim == 1:
			self.labels     = ["Dist. [pc]"]
			prefix          = "dist"
			self.labels_csv = [ prefix+su for su in suffix]
			if self.transformation is not None :
				print("For 1D version there is no transformation")

		elif n_dim == 3:
			if self.transformation is not None:
				self.labels = ["X [pc]","Y [pc]","Z [pc]"]
				prefix      = ["X","Y","Z"]
			else:
				self.labels     = ["R.A. [deg]","Dec. [deg]","Dist. [pc]"]
				prefix          = ["ra","dec","dist"]

			self.labels_csv = sum([[pre+su for su in suffix] for pre in prefix],[])

		elif n_dim == 5:
			self.labels     = ["R.A. [deg]","Dec. [deg]","Dist. [pc]","pmra [mas/yr]","pmdec [mas/yr]"]
			prefix          = ["ra","dec","dist","pmra","pmdec"]
			self.labels_csv = sum([[pre+su for su in suffix] for pre in prefix],[])
			if self.transformation is not None :
				print("For 5D version there is no transformation")

		elif n_dim == 6:
			if self.transformation is not None:
				self.labels = ["X [pc]","Y [pc]","Z [pc]","U [km/s]","V [km/s]","W [km/s]"]
				prefix      = ["X","Y","Z","U","V","W"]
			else:
				self.labels     = ["R.A. [deg]","Dec. [deg]","Dist. [pc]","pmra [mas/yr]","pmdec [mas/yr]","Rvel [km/s]"]
				prefix          = ["ra","dec","dist","pmra","pmdec","rvel"]

			self.labels_csv = sum([[pre+su for su in suffix] for pre in prefix],[])

		else:
			sys.exit("Incorrect dimension")

		#=================================================================================

		
		with h5py.File(file_name, 'r') as hf:
			if names is not None :
				if names not in list(hf.keys()):
					RuntimeError("{0} not in {1}".format(names,file_name))
					sys.exit(0)
			else:
				names = list(hf.keys())

			self.names = sorted(names,key=lambda x:int(x))

		#---------- Plots -------------------
		if not os.path.isdir(dir_plots):
			os.mkdir(dir_plots)

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		#------------------------------------------

		#--------- Verify convergence ----------------------------
		conv = self.convergence(tol_convergence)
		if np.logical_not(conv).any():
			warnings.warn("File {0} contains unconverged chains".format(self.file_name),Warning)
		#---------------------------------------------------------


		#---------- Initialize the Rotation ----------
		if self.transformation is not None:
			if self.transformation is "ICRS2GAL" and n_dim is 3:
				CoordTrans = CoordinateTransformation(Transformations.ICRS2GAL)
				self.Rotation = CoordTrans.transformCartesianCoordinates
			elif self.transformation is "ICRS2ECL" and n_dim is 3:
				CoordTrans = CoordinateTransformation(Transformations.ICRS2ECL)
				self.Rotation = CoordTrans.transformCartesianCoordinates
			elif self.transformation is "XYZ" and n_dim in [3,6]:
				self.Rotation = Identity3D
			else:
				sys.exit("Transformation not valid!")
			

		
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

	def Transform(self,sample):
		#-------------- Properties of chain -----------------
		N,walkers,D = sample.shape
		assert self.n_dim == D, "The dimension in the chain differs from that specified!"
		#----------------------------------------------------------

		if self.transformation is None or self.n_dim in [1,5] :
			return sample

		sample = np.reshape(sample,(N*walkers,D))

		if self.n_dim == 3:
			sample[:,0] = np.radians(sample[:,0])
			sample[:,1] = np.radians(sample[:,1])
			X,Y,Z = sphericalToCartesian(sample[:,2],sample[:,0],sample[:,1])
			X,Y,Z = self.Rotation(X,Y,Z)
			sample = np.column_stack((X,Y,Z))

		if self.n_dim == 6:
			sample[:,0] = np.radians(sample[:,0])
			sample[:,1] = np.radians(sample[:,1])
			sample[:,2] = 1000.0/sample[:,2] # Back to parallax in mas :'(
			# This need to be done to use the following function
			X,Y,Z,U,V,W = astrometryToPhaseSpace(sample[:,0],sample[:,1],sample[:,2],
												 sample[:,3],sample[:,4],sample[:,5])
			sample = np.column_stack((X,Y,Z,U,V,W))

		sample = np.reshape(sample,(N,walkers,D))
		return sample

	def plot_chains(self,names=None,true_values=None,title_fmt=".2f"):
		"""
		This function plots the chains and the marginals

		Arguments:
		true_values (float): Numpy array with true values.
		"""
		if names is None:
			names = self.names

		if true_values is not None:
			n_t,d_t = true_values.shape
			assert n_t == len(names), 'True values must be of same length as names'
			assert d_t == self.n_dim, 'True values must be of same dimension as parameters'
		else:
			true_values = np.repeat(None,len(names))

		#---------- Loop over names in file ---------------------------
		tbar = tqdm(total=len(names),desc="Plotting: ")
		for i,name in enumerate(names):
			reader = emcee.backends.HDFBackend(self.file_name,name=name)
			tau    = reader.get_autocorr_time(tol=0)
			burnin = int(self.burnin_tau*np.max(tau))
			sample = reader.get_chain(discard=burnin)

			#------------- Transform ----------------------------------
			sample = self.Transform(sample)	 
			#----------------------------------------------------------

			#--------- MAP --------------------------------------------
			logpro   = reader.get_log_prob(discard=burnin)
			idx_map  = np.unravel_index(logpro.argmax(), logpro.shape)
			MAP      = sample[idx_map]
			#----------------------------------------------------------

			pdf = PdfPages(filename=self.dir_plots+str(self.n_dim)+"D_"+self.id+"_"+str(name)+".pdf")
			plt.figure(1)
			#----------- Trace plots --------------------------
			if self.n_dim == 1:
				plt.plot(sample[:, :, 0], "k", alpha=0.3)
				plt.ylabel(self.labels[0])
				plt.xlabel("Step")

			else :
				fig, axes = plt.subplots(self.n_dim, figsize=self.figure_size, sharex=True)
				for j in range(self.n_dim):
				    ax = axes[j]
				    ax.plot(sample[:, :, j], "k", alpha=0.3)
				    ax.set_ylabel(self.labels[j])
				    ax.yaxis.set_label_coords(-0.1, 0.5)

				axes[-1].set_xlabel("Step")

			#-------------- Save fig --------------------------
			pdf.savefig(bbox_inches='tight')
			plt.close(1)
			
			#----------- Corner plot --------------------------
			sample = sample.reshape((-1,self.n_dim))
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
				if self.statistic is "map":
					q_ctr = MAP[i]
				q_m, q_p = q_ctr-q_low, q_max-q_ctr

				# Format the quantile display.
				fmt = "{{0:{0}}}".format(title_fmt).format
				title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
				title = title.format(fmt(q_ctr), fmt(q_m), fmt(q_p))
				ax.set_title(title)
				ax.axvline(q_low, color="k",linestyle=":")
				ax.axvline(q_ctr, color="k",linestyle="--")
				ax.axvline(q_max, color="k",linestyle=":")
			#===========================================================

			fig.set_size_inches(self.figure_size)
			#----------------------------------------------
			pdf.savefig(bbox_inches='tight')
			plt.close("all")
			pdf.close()
			tbar.update(1)
		tbar.close()

	def get_statistics(self,names):
		"""
		This function computes the chain statistics.

		Arguments:
		Returns (array):    For each parameter it contains min,central, and max. 
		"""

		stats = np.zeros((len(names),self.n_dim*3))
		#---------- Loop over names in file ---------------------------
		tbar = tqdm(total=len(names),desc="Computing statistics: ")
		for i,name in enumerate(names):
			reader = emcee.backends.HDFBackend(self.file_name,name=name)
			tau    = reader.get_autocorr_time(tol=0)
			burnin = int(self.burnin_tau*np.max(tau))
			sample = reader.get_chain(discard=burnin)

			#------------- Transform ----------------------------------
			sample = self.Transform(sample)
			#----------------------------------------------------------

			#--------- MAP --------------------------------------------
			logpro   = reader.get_log_prob(discard=burnin)
			idx_map  = np.unravel_index(logpro.argmax(), logpro.shape)
			MAP      = sample[idx_map]
			#----------------------------------------------------------

			for d in range(self.n_dim):
				stats[i,(3*d):(3*d + 3)] = self.statistics(sample[:,:,d])

			if self.statistic is "map":
				if self.n_dim == 1:
					stats[i,1]  = MAP[0]
				elif self.n_dim == 3:
					stats[i,[1,4,7]]  = MAP
				elif self.n_dim == 5:
					stats[i,[1,4,7,10,13]]  = MAP
				elif self.n_dim == 6:
					stats[i,[1,4,7,10,13,16]]  = MAP
				else:
					print("Not yet implemented!")

			tbar.update(1)
		tbar.close()
		
		return stats


	def statistics(self,sample):
		'''
		Computes the statistics of the parameter
		'''
		sts = np.zeros(3)
		sts[[0,2]] = np.quantile(sample,self.quantiles)
		if  self.statistic is "median":
			ctr = np.median(sample)
		elif self.statistic is "mean":
			ctr = np.mean(sample)
		elif self.statistic is "mode":
			mins,maxs = np.min(sample),np.max(sample)
			x         = np.linspace(mins,maxs,num=self.npts_mode)
			gkde      = st.gaussian_kde(sample.flatten())
			ctr       = x[np.argmax(gkde(x))]
		elif self.statistic is "map":
			# This will be populated in the get_statistics function
			ctr = 0.0
		else:
			sys.exit("method not defined")

		sts[1] = ctr

		return sts


	def save_statistics(self,file_csv,names=None):
		'''
		Saves the statistics to a csv file.
		Arguments:
		file_csv (string) the name of the file where to save the statistics
		names (list of strings) the names of the objects

		Note: The statistic name will be appended.
		'''

		if names is None:
			names = self.names
		#-------------- Adds the statistic to the file name ------------
		file_csv = file_csv.replace(".csv","_"+self.statistic+".csv")

		if self.transformation is not None:
			if self.n_dim == 3:
				file_csv = file_csv.replace(".csv","_XYZ.csv")
			elif self.n_dim == 6:
				file_csv = file_csv.replace(".csv","_XYZUVW.csv")
			else:
				print("Error in dimension")


		stats = self.get_statistics(names=names)

		df_stats = pn.DataFrame(data=stats,columns=self.labels_csv)
		df_id    = pn.DataFrame(data=names,columns=[self.id])

		df       = df_id.join(df_stats)
		df.to_csv(path_or_buf=file_csv,index=False)


