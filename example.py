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
os.environ["MKL_NUM_THREADS"] = "1" # Avoids overlapping of processes
os.environ["OMP_NUM_THREADS"] = "1" # Avoids overlapping of processes
import numpy as np
import h5py

#----- Import the module -------------------------------
dir_kalkayotl  = "/home/jolivares/Repos/Kalkayotl/" 
sys.path.append(dir_kalkayotl)
from kalkayotl.inference import Inference
#-------------------------------------------------------

#============ Directory and data ===========================================
dir_base = "/home/jolivares/Repos/Kalkayotl/article/v2.0/ComaBer/Core/pymc5/"

#----------- Data file -----------------------------------------------------
file_data = dir_base + "members+rvs.csv"
file_parameters = None
#----------------------------------------------------------------------------

#------- Creates directory if it does not exists -------
os.makedirs(dir_base,exist_ok=True)
#-------------------------------------------------------
#============================================================================


#=============== Tuning knobs ============================
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
tuning_iters = 3000

# After discarding the burning you will obtain sample_iters*chains samples
# from the posterior distribution. These are the ones used in the plots and to
# compute statistics.
sample_iters = 3000


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
reference_system = "Galactic"

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
#-----------------------------------------------------------------------------------------

#----------- Velocity model --------------------------
# Different types of velocity models are implemented:
# "join": this is the most general which results in a joint model in position+velocity
# "independent": independently models positions and velocities.
# "constant": models the velocity as expanding or contracting field
# "linear": models the velocity field as a linear function of position.
#----------------------------------------------------------------------------------------

nuts_backend = "numpyro"
#=========================================================================================

#========================= PRIORS ===========================================
list_of_prior = [
	{"type":"Gaussian",
		"dimension":dimension,
		"zero_point":zero_point[:dimension],
		"parameters":{"location":None,"scale":None},
		"hyper_parameters":{
							"alpha":None,
							"beta":None,
							"gamma":None,
							"delta":None,
							"eta":None
							},
		"field_sd":None,
		"parametrization":"central",
		"velocity_model":"joint",
		"optimize":True},
	# {"type":"StudentT",
	# 	"dimension":dimension,
	# 	"zero_point":zero_point[:dimension],
	# 	"parameters":{"location":None,"scale":None},
	# 	"hyper_parameters":{
	# 						"alpha":None,
	# 						"beta":None,
	# 						"gamma":None,
	# 						"delta":None,
	# 						"eta":None,
	# 						"nu":None,
	# 						},
	# 	"field_sd":None,
	# 	"parametrization":"central",
	# 	"velocity_model":"joint",
	# 	"optimize":False},
	# {"type":"FGMM",
	# 	"dimension":dimension,
	# 	"zero_point":zero_point[:dimension],        
	# 	"parameters":{"location":None,
	# 				  "scale":None,
	# 				  "weights":None},
	# 	"hyper_parameters":{
	# 						"alpha":None,
	# 						"beta":None, 
	# 						"delta":[1,1], # Field in the second entry
	# 						"eta":None,
	# 						},
	# 	"field_sd":{"position":50.0,"velocity":10.0},
	# 	"parametrization":"central",
	# 	"velocity_model":"joint",
	# 	"optimize":False},
	# {"type":"CGMM",
	# 	"dimension":dimension,
	# 	"zero_point":zero_point[:dimension],        
	# 	"parameters":{"location":None,
	# 				  "scale":None,
	# 				  "weights":None},
	# 	"hyper_parameters":{
	# 						"alpha":None,
	# 						"beta":None, 
	# 						"gamma":None,
	# 						"delta":np.repeat(2,2),
	# 						"eta":None,
	# 						"n_components":2
	# 						},
	# 	"field_sd":None,
	# 	"parametrization":"central",
	# 	"velocity_model":"joint",
	# 	"optimize":False},

	# {"type":"GMM",
	# 	"dimension":dimension,
	# 	"zero_point":zero_point[:dimension],        
	# 	"parameters":{"location":file_parameters,
	# 				  "scale":file_parameters,
	# 				  "weights":file_parameters},
	# 	"hyper_parameters":{
	# 						"alpha":None,
	# 						"beta":None, 
	# 						"gamma":None,
	# 						"delta":np.repeat(1,2),
	# 						"eta":None,
	# 						"n_components":2
	# 						},
	# 	"field_sd":None,
	# 	"parametrization":"central",
	# 	"velocity_model":"joint",
	# 	"optimize":False}
	]
#======================= Inference and Analysis =====================================================

#--------------------- Loop over prior types ------------------------------------
for prior in list_of_prior:

	#------ Output directories for each prior -------------------
	dir_prior = dir_base +  "{0}D_{1}_{2}_{3}_{4}_{5}".format(
		dimension,
		prior["type"],
		reference_system,
		prior["parametrization"],
		prior["velocity_model"],
		nuts_backend)
	#------------------------------------------------------------

	#---------- Create prior directory -------------
	os.makedirs(dir_prior,exist_ok=True)
	#------------------------------------------------

	#--------- Initialize the inference module -------
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
			  parametrization=prior["parametrization"],
			  field_sd=prior["field_sd"],
			  velocity_model=prior["velocity_model"])

	#============ Sampling with HMC ======================================
	#------- Run the sampler ---------------------
	p3d.run(sample_iters=sample_iters,
			tuning_iters=tuning_iters,
			target_accept=target_accept,
			optimize=prior["optimize"],
			chains=chains,
			cores=cores,
			nuts_backend=nuts_backend)
	#-------------------------------------

	# -------- Load the chains --------------------------------
	# This is useful if you have already computed the chains
	# and want to re-analyse (in that case comment the p1d.run() line)
	p3d.load_trace()

	# ------- Re-analyse the convergence of the sampler---
	p3d.convergence()

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
	
#=======================================================================================
