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
#------------ Load libraries -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
import numpy as np

#----- Import the module ----------
from kalkayotl import Inference


#============ Data and Directories =============================
#-------Main directory ---------------
dir_main  = os.getcwd() + "/"
#-------------------------------------

#------------------------- Cluster ----------------------------
case      = "Ruprecht_147"     # Case name
file_csv  = "Ruprecht_147.csv" # File name of the data set
#-----------------------------------------------------------

#----------- Data file --------------------
file_data = dir_main + "Data/" + file_csv
#-----------------------------------------

#--------- Directories where chains and plots will be saved ----
dir_out    = dir_main + "Test/"
dir_case   = dir_out  + case +"/"
#--------------------------------------

#------- Creates directories -------
os.makedirs(dir_out,exist_ok=True)
os.makedirs(dir_case,exist_ok=True)
#---------------------------------

#==================================================


#=============== Tuning knobs ============================
# --------- Transformation------------------------------------
# In which space you want to sample: distance or parallax?
# For distance use "pc", for parallax use "mas"
# IMPORTANT: The units of the parameters and hyper-parameters
# defined below must coincide with those of the chosen transformation.
transformation = "pc"

#----------------- Chains-----------------------------------------------------
# The number of parallel chains you want to run. Two are the minimum required
# to analyse convergence.
chains = 2

# Number of computer cores to be used. You can increase it to run faster.
# IMPORTANT. Depending on your computer configuration you may have different performances.
# I recommend to use 2 cores; this is one per chain.
cores  = 2

# burining_iters is the number of iterations used to tune the sampler
# These will not be used for the statistics nor the plots. 
# If the sampler shows warnings you most probably must increase this value.
burning_iters   = 1000  

# After discarding the burning you will obtain sample_iters*chains samples
# from the posterior distribution. These are the ones used in the plots and to
# compute statistics.
sample_iters    = 1000
#---------------------------------------------------------------------------


#------------ Statistic ----------------------------------------------
# Chose your favourite statistic and quantiles.
# This will be computed and written in both 
# Source_{statistic}.csv and Cluster_{statistic}.csv files
statistic = "mode"
quantiles = [0.05,0.95]
#----------------------------------------------------------------------


#--------- Zero point -----------------------------------------------
# The zero point of the parallax measurements
# You can provide either a scalar or a vector of the same dimension
# as the valid sources in your data set.
zero_point = -0.029  # This is Lindegren+208 value
#---------------------------------------------------------------------

#------- Independent measurements--------
# In the Gaia astrometric data the measurements of stars are correlated between sources.
# By default, Kalkayotl will not assume independence amongst sources.
# Set it to True if you want to assume independence, 
# and thus neglect the parallax spatial correlations. 
indep_measures = False
#==========================================================



#============= Prior and hyper-parameters ================================================
# The following is a list of priors with different parameters and
# hyper-parameters. 

# parameters is a dictionary with two entries: "location" and "scale".
# For each of them you can either provide a value or set it to None to infer it.
# Notice that you can infer one or both.
# IMPORTANT. In the EDSD prior you must set both. Location to zero and
# scale to the scale length (in pc).

#----- Hyper-parameters --------------------------------------------
# hyper_alpha controls the cluster location, which is Gaussian distributed.
# Therefore you need to specify the median and standard deviation, in that order.
hyper_alpha = [305,30.5]

# hyper_beta controls  the cluster scale, which is Gamma distributed.
# hyper_beta corresponds to the mode of the distribution.
hyper_beta = [20.]

# hyper_gamma controls the gamma and tidal radius parameters in 
# the EFF and King priors. Set it to None in other priors.
# The EFF uses a truncated Gaussian as prior, thus you must specify 
# the mean and standard deviations. It has no units.
# The King uses a Half Gaussian thus it only uses the first value as
# standard deviation. It is scaled to the cluster scale.

# hyper_delta is only used in the GMM prior (use None in the rest),
# where it represents the vector of hyper-parameters for the Dirichlet
# distribution controlling the weights in the mixture.
# IMPORTANT. The number of Gaussians in the mixture corresponds to the
# length of this vector. 


list_of_prior = [
	# {"type":"EDSD",         "parameters":{"location":0.0,"scale":1350.0}, 
	# 						"hyper_alpha":None, 
	# 						"hyper_beta":None, 
	# 						"hyper_gamma":None,
	# 						"hyper_delta": None,
							# "burning_factor":1},

	# {"type":"Uniform",      "parameters":{"location":None,"scale":None},
	# 						"hyper_alpha":hyper_alpha,
	# 						"hyper_beta":hyper_beta,
	# 						"hyper_gamma":None, 
	# 						"hyper_delta":None,
							# "burning_factor":1},

	# {"type":"Gaussian",     "parameters":{"location":None,"scale":None},
	# 						"hyper_alpha":hyper_alpha,
	# 						"hyper_beta":hyper_beta,
	# 						"hyper_gamma":None,
	# 						"hyper_delta":None,
							# "burning_factor":1},

	# {"type":"King",         "parameters":{"location":None,"scale":None,"rt":None},
	# 						"hyper_alpha":hyper_alpha, 
	# 						"hyper_beta":hyper_beta, 
	# 						"hyper_gamma":[10.0,1.0],
	# 						"hyper_delta":None,
	# 						"burning_factor":2},

	# {"type":"EFF",          "parameters":{"location":None,"scale":None,"gamma":None},
	# 						"hyper_alpha":hyper_alpha,
	# 						"hyper_beta":hyper_beta, 
	# 						"hyper_gamma":[3.0,1.0],
	# 						"hyper_delta":None,
	# 						"burning_factor":2},

	{"type":"GMM",          "parameters":{"location":None,"scale":None,"weights":None},
							"hyper_alpha":hyper_alpha, 
							"hyper_beta":hyper_beta, 
							"hyper_gamma":None,
							"hyper_delta":np.array([0.5,0.5]),
							"burning_factor":10}

	]


#======================= Inference and Analysis =====================================================

#--------------------- Loop over prior types ------------------------------------
for prior in list_of_prior:

	#------ Output directories for each prior -------------------
	dir_prior = dir_case + prior["type"]
	dir_out   = dir_prior + "/" 

	os.makedirs(dir_prior,exist_ok=True)
	os.makedirs(dir_out,exist_ok=True)
	#------------------------------------------------

	#--------- Initialize the inference module ----------------------------------------
	p1d = Inference(dimension=1,                       # For now it only works in 1D.
					prior=prior["type"],
					parameters=prior["parameters"],
					hyper_alpha=prior["hyper_alpha"],
					hyper_beta=prior["hyper_beta"],
					hyper_gamma=prior["hyper_gamma"],
					hyper_delta=prior["hyper_delta"],
					dir_out=dir_out,
					transformation=transformation,
					zero_point=zero_point,
					indep_measures=indep_measures)
	#-------- Load the data set --------------------
	# It will use the Gaia column names by default.
	p1d.load_data(file_data)

	#------ Prepares the model -------------------
	# p1d.setup()

	#------- Run the sampler ---------------------
	# p1d.run(sample_iters=sample_iters*prior["burning_factor"],
	# 		burning_iters=burning_iters,
	# 		chains=chains,
	# 		cores=cores)

	# -------- Load the chains --------------------------------
	# This is useful if you have already computed the chains
	# and want to re-analyse (in that case comment the p1d.run() line)
	# p1d.load_trace(sample_iters=sample_iters)

	# ------- Re-analyse the convergence of the sampler---
	# p1d.convergence()

	#-------- Plot the trace of the chains -------------
	# If you provide the list of IDs (string list) it will plot the traces
	# of the provided sources.
	# p1d.plot_chains(dir_out,IDs=['4087735025198194176'])

	#----- Compute and save the posterior statistics ---------
	# p1d.save_statistics(statistic=statistic,quantiles=quantiles)

	#------- Save the samples into HDF5 file --------------
	# p1d.save_samples()

	#----------------------- Evidence --------------------
	# Uncomment if you want to compute the evidence.
	# IMPORTANT. It may take a long time to compute.

	# Output file, it will contain the logarithm of the evidence.
	# and noisy and inaccurate estimates of the parameters.
	file_Z    = dir_out   + "Cluster_Z.csv"

	# N_samples is the number of sources from the data set that will be used
	# Set to None to use all sources

	# M_samples is the number of samples to draw from the prior. The larger the better
	# but it will further increase computing time.

	# dlogz is the tolerance in the evidence computation 
	# The sampler will stop once this value is attained.

	# nlive is the number of live points used in the computation. The larger the better
	# but it will further increase the computing time.

	p1d.evidence(M_samples=1000,dlogz=1.0,nlive=100,file=file_Z,print_progress=True)
	#----------------------------------------------------------------------------------
#=======================================================================================