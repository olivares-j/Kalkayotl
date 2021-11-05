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
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np
import h5py

#----- Import the module ----------
from kalkayotl import Inference
from kalkayotl.Transformations import astrometryToPhaseSpace


part = 5
#============ Directory and data ===========================================
dir_main = "/home/jromero/OCs/USco/Kalkayotl/Classification/"

#----- Directory where chains and plots will be saved ----
dir_out  = dir_main + "part_{0}/".format(part)
#-------------------------------------------------------------------------

#----------- Data file -----------------------------------------------------
file_data = dir_out + "members_{0}.csv".format(part)
file_parameters = dir_main + "Cluster_eight_pop.csv"
#----------------------------------------------------------------------------

#------- Creates directory if it does not exists -------
os.makedirs(dir_out,exist_ok=True)
#-------------------------------------------------------
#============================================================================


#=============== Tuning knobs ============================
n_gaussians = 8
dimension = 6
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
tuning_iters = 1000

# After discarding the burning you will obtain sample_iters*chains samples
# from the posterior distribution. These are the ones used in the plots and to
# compute statistics.
sample_iters = 1000


#----- Target_accept-------
# This parameter controls the acceptance of the proposed steps in the Hamiltonian
# Monte Carlo sampler. It should be larger than 0.7-0.8. Increasing it helps in the convergence
# of the sampler but increases the computing time.
target_accept = 0.95
#---------------------------------------------------------------------------

#------------ Statistic -------------------------------------------------------
# mean, median and mode together with high density interval will be computed
# The outputs will be at Source_statistics.csv and Cluster_statistics.csv files
hdi_prob = 0.95
#------------------------------------------------------------------------------

# --------- Transformation------------------------------------
# In which space you want to sample: distance or parallax?
# For distance use "pc", for parallax use "mas"
# IMPORTANT: The units of the parameters and hyper-parameters
# defined below must coincide with those of the chosen transformation.
transformation = "pc"

#------------- Reference system -----------
# Coordinate system in which parameters will be inferred
# Either "ICRS" or "Galactic"
reference_system = "ICRS"

#--------- Zero point -----------------------------------------------
# The zero point of the parallax measurements
# You can provide either a scalar or a vector of the same dimension
# as the valid sources in your data set.
zero_point = [0.,0.,-0.017,0.,0.,0.]  # This is Brrowns+2020 value
#---------------------------------------------------------------------

#------- Independent measurements--------
# In the Gaia astrometric data the measurements of stars are correlated between sources.
# By default, Kalkayotl will not assume independence amongst sources.
# Set it to True if you want to assume independence, 
# and thus neglect the parallax spatial correlations. 
indep_measures = False

#------ Parametrization -----------------
# The performance of the HMC sampler can be improved by non-central parametrizations.
# Kalkayotl comes with two options: central and non-central. While the former works better
# for nearby clusters (<500 pc) the latter does it for faraway clusters (>500 pc).
#==========================================================

#========================= PRIORS ===========================================
list_of_prior = [
	# {"type":"Gaussian",
	# 	"dimension":dimension,
	# 	"zero_point":zero_point[:dimension],
	# 	"parameters":{"location":None,"scale":None},
	# 	"hyper_parameters":{
	# 						"alpha":hyper_alpha[:dimension],
	# 						"beta":hyper_beta,
	# 						"gamma":None,
	# 						"delta":None,
	# 						"eta":hyper_eta
	# 						},
	# 	"parametrization":"central",
	# 	"prior_predictive":False,
	# 	"optimize":True},

	# {"type":"Gaussian",
	# 	"dimension":dimension,
	# 	"zero_point":zero_point[:dimension],
	# 	"parameters":{"location":None,"scale":None},
	# 	"hyper_parameters":{
	# 						"alpha":hyper_alpha[:dimension],
	# 						"beta":hyper_beta,
	# 						"gamma":None,
	# 						"delta":None,
	# 						"eta":hyper_eta
	# 						},
	# 	"parametrization":"non-central",
	# 	"prior_predictive":False,
	# 	"optimize":False},

	
	# {"type":"King",
	# 	"dimension":dimension,
	# 	"zero_point":zero_point[:dimension],     
	# 	"parameters":{"location":None,"scale":None,"rt":None},
	# 	"hyper_parameters":{
	# 						"alpha":hyper_alpha[:dimension], 
	# 						"beta":hyper_beta, 
	# 						"gamma":10.0,
	# 						"delta":None,
	# 						"eta":hyper_eta
	# 						},
	# 	"parametrization":"non-central",
	# 	"prior_predictive":False,
	# 	"optimize":False},
	# # # NOTE: the tidal radius and its parameters are scaled.

	
	# {"type":"EFF",
	# 	"dimension":dimension,
	# 	"zero_point":zero_point[:dimension],      
	# 	"parameters":{"location":None,"scale":None,"gamma":None},
	# 	"hyper_parameters":{
	# 						"alpha":hyper_alpha[:dimension],
	# 						"beta":hyper_beta, 
	# 						"gamma":2.0,
	# 						"delta":None,
	# 						"eta":hyper_eta
	# 						},
	# 	"parametrization":"non-central",
	# 	"prior_predictive":False,
	# 	"optimize":False},
	# # NOTE: the mode of the Gamma parameter will be at 3.0 + hyper_gamma

	{"type":"GMM",
		"dimension":dimension,
		"zero_point":zero_point[:dimension],        
		"parameters":{"location":file_parameters,
					  "scale":file_parameters,
					  "weights":None},
		"hyper_parameters":{
							"alpha":None,
							"beta":50.0, 
							"gamma":None,
							"delta":np.repeat(2,n_gaussians),
							"eta":None,
							"n_components":n_gaussians
							},
		"parametrization":"central",
		"optimize":True},

	# {"type":"CGMM",
	# 	"dimension":dimension,
	# 	"zero_point":zero_point[:dimension],       
	# 	"parameters":{"location":None,"scale":None,"weights":None},
	# 	"hyper_parameters":{
	# 						"alpha":hyper_alpha[:dimension], 
	# 						"beta":hyper_beta, 
	# 						"gamma":None,
	# 						"delta":np.array([5,5]),
	# 						"eta":hyper_eta
	# 						},
	# 	"parametrization":"central",
	# 	"prior_predictive":False,
	# 	"optimize":False}
	]
#======================= Inference and Analysis =====================================================

#--------------------- Loop over prior types ------------------------------------
for prior in list_of_prior:

	#------ Output directories for each prior -------------------
	dir_prior = dir_out + str(dimension) + "D_" + str(n_gaussians) + prior["type"] + "_" + prior["parametrization"]

	#---------- Create prior directory -------------
	os.makedirs(dir_prior,exist_ok=True)
	#------------------------------------------------

	#--------- Initialize the inference module ----------------------------------------
	p3d = Inference(dimension=prior["dimension"],
					dir_out=dir_prior,
					zero_point=prior["zero_point"],
					indep_measures=indep_measures,
					reference_system=reference_system)

	#-------- Load the data set --------------------
	# It will use the Gaia column names by default.
	p3d.load_data(file_data)

	#------ Prepares the model -------------------
	p3d.setup(prior=prior["type"],
			  parameters=prior["parameters"],
			  hyper_parameters=prior["hyper_parameters"],
			  transformation=transformation,
			  parametrization=prior["parametrization"])

	#============ Sampling with HMC ======================================
	#------- Run the sampler ---------------------
	# p3d.run(sample_iters=sample_iters,
	# 		tuning_iters=tuning_iters,
	# 		target_accept=target_accept,
	# 		optimize=prior["optimize"],
	# 		chains=chains,
	# 		cores=cores)

	# -------- Load the chains --------------------------------
	# This is useful if you have already computed the chains
	# and want to re-analyse (in that case comment the p1d.run() line)
	p3d.load_trace()

	# ------- Re-analyse the convergence of the sampler---
	# p3d.convergence()

	#-------- Plot the trace of the chains ------------------------------------
	# If you provide the list of IDs (string list) it will plot the traces
	# of the provided sources. If IDs keyword removed only plots the population parameters.
	p3d.plot_chains()

	#------- Plot model ----------------
	p3d.plot_model()

	#----- Compute and save the posterior statistics ---------
	p3d.save_statistics(hdi_prob=hdi_prob)

	#------- Save the samples into HDF5 file --------------
	p3d.save_samples()

	#=============== Evidence computation ==============================================
	# IMPORTANT. It will increase the computing time!

	# N_samples is the number of sources from the data set that will be used
	# Set to None to use all sources

	# M_samples is the number of samples to draw from the prior. The larger the better
	# but it will further increase computing time.

	# dlogz is the tolerance in the evidence computation 
	# The sampler will stop once this value is attained.

	# nlive is the number of live points used in the computation. The larger the better
	# but it will further increase the computing time.

	# UNCOMMENT NEXT LINE
	# p1d.evidence(M_samples=1000,dlogz=1.0,nlive=100)
	#----------------------------------------------------------------------------------
	#===================================================================================

	#=============== Extract Samples =========================================
	# file_samples = dir_prior + "/Samples.h5"
	# hf = h5py.File(file_samples,'r')

	# #---------------- Sources -----------------------------------------
	# srcs = hf.get("Sources")

	# n_samples = 100
	# samples = np.empty((len(srcs.keys()),dimension,n_samples))
	# #-------- loop over array and fill it with samples -------
	# for i,ID in enumerate(srcs.keys()):
	# 	#--- Extracts a random choice of the samples --------------
	# 	tmp = np.array(srcs.get(str(ID)))
	# 	idx = np.random.choice(np.arange(tmp.shape[1]),size=n_samples,
	# 							replace=False)
	# 	samples[i] = tmp[:,idx]
	# 	#----------------------------------------------------------

	# 	distance = np.sqrt(np.sum(samples[i]**2,axis=0))
	# 	print("Source {0} at {1:3.1f} +/- {2:3.1f} pc.".format(ID,
	# 										distance.mean(),
	# 										distance.std()))
	# #--------------------------------------------------------------------

	# #-------------- Cluster --------------------------------------------
	# cluster = hf.get("Cluster")
	# loc = np.empty((3,sample_iters*chains))
	# for i in range(3):
	# 	loc[i] = np.array(cluster.get("3D_loc_{0}".format(i))).flatten()

	# distance = np.sqrt(np.sum(loc**2,axis=0))

	# print("The cluster distance is {0:3.1f} +/- {1:3.1f}".format(np.mean(distance),
	# 															np.std(distance)))

	# #- Close HDF5 file ---
	# hf.close()
	# #============================================================================
#=======================================================================================


