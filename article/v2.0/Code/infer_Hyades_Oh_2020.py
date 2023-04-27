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
dir_base = "/home/jolivares/Repos/Kalkayotl/article/v2.0/Hyades/Oh_2020/Core/"

#----------- Data file -----------------------------------------------------
file_data = dir_base + "members_core_GDR2.csv"
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
tuning_iters = 1000
sample_iters = 1000
target_accept = 0.95

sampling_space = "physical"
reference_system = "ICRS"

#--------- Zero point ------------
# A dcictionary with zerpoints
zero_points = {
"ra":0.,
"dec":0.,
"parallax":0.,
"pmra":0.,
"pmdec":0.,
"radial_velocity":0.}  
#--------------------------------

indep_measures = True
velocity_model = "joint"
nuts_sampler = "numpyro"

#=========================================================================================

#========================= PRIORS ===========================================
prior = {"type":"Gaussian",
		"parameters":{"location":None,"scale":None},
		"hyper_parameters":{
							"alpha":None,
							"beta":None,
							"gamma":None,
							"delta":None,
							"eta":None
							},
		"parametrization":"central"}
#======================= Inference and Analysis =====================================================

#------ Output directories for each prior -------------------
dir_prior = dir_base +  "{0}D_{1}_{2}_{3}_indep{4}_{5}".format(
	dimension,
	prior["type"],
	reference_system,
	velocity_model,
	indep_measures,
	nuts_sampler)
#------------------------------------------------------------

#---------- Create prior directory -------------
os.makedirs(dir_prior,exist_ok=True)
#------------------------------------------------

#--------- Initialize the inference module -------
kal = Inference(dimension=dimension,
				dir_out=dir_prior,
				zero_points=zero_points,
				indep_measures=indep_measures,
				reference_system=reference_system)

#-------- Load the data set --------------------
# It will use the Gaia column names by default.
kal.load_data(file_data,corr_func="Lindegren+2018",radec_precision_arcsec=5.0,)

#------ Prepares the model -------------------
kal.setup(prior=prior["type"],
		  parameters=prior["parameters"],
		  hyper_parameters=prior["hyper_parameters"],
		  parametrization=prior["parametrization"],
		  sampling_space=sampling_space,
		  velocity_model=velocity_model)

#============ Sampling with HMC ======================================
#------- Run the sampler ---------------------
kal.run(sample_iters=sample_iters,
		tuning_iters=tuning_iters,
		target_accept=target_accept,
		chains=chains,
		cores=cores,
		init_iters=int(1e5),
		init_refine=True,
		nuts_sampler=nuts_sampler,
		posterior_predictive=True,
		prior_predictive=True)
#-------------------------------------

# -------- Load the chains --------------------------------
# This is useful if you have already computed the chains
# and want to re-analyse (in that case comment the p1d.run() line)
kal.load_trace()

# ------- Re-analyse the convergence of the sampler---
kal.convergence()

#-------- Plot the trace of the chains ------------------------------------
# If you provide the list of IDs (string list) it will plot the traces
# of the provided sources. If IDs keyword removed only plots the population parameters.
kal.plot_chains()

#--- Check Prior and Posterior ----
kal.plot_prior_check()
#--------------------------------

#--- Plot model -- 
kal.plot_model()
# -----------------

#----- Compute and save the posterior statistics ---------
kal.save_statistics(hdi_prob=hdi_prob)

kal.save_posterior_predictive()

#------- Save the samples into HDF5 file --------------
kal.save_samples()
#=======================================================================================
