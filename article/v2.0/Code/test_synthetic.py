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
dir_base = "/home/jolivares/Repos/Kalkayotl/article/v2.0/Synthetic/Gaussian_linear/"

#----------- Data file -----------------------------------------------------
file_data = dir_base + "Gaussian_n50.csv"
#----------------------------------------------------------------------------

#------- Creates directory if it does not exists -------
os.makedirs(dir_base,exist_ok=True)
#-------------------------------------------------------
#============================================================================

#=============== Tuning knobs ============================
dimension = 6
chains = 2
cores  = 2
tuning_iters = 4000
sample_iters = 2000
sampling_space = "physical"
reference_system = "Galactic"
radec_precision_arcsec = 1.0

#--------- Zero point ------------
# A dcictionary with zerpoints
zero_points = {
"ra":0.,
"dec":0.,
"parallax":0.0,
"pmra":0.,
"pmdec":0.,
"radial_velocity":0.}  
#--------------------------------

indep_measures = False
velocity_model = "linear"
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
		"parametrization":"non-central"}
#======================= Inference and Analysis =====================================================

#--------------------- Loop over prior types ------------------------------------
for target_accept in [0.6]:

	#------ Output directories for each prior -------------------
	dir_prior = dir_base +  "{0}D_{1}_{2}_{3}_{4}_{5}".format(
		dimension,
		prior["type"],
		prior["parametrization"],
		velocity_model,
		radec_precision_arcsec,
		target_accept)
	#------------------------------------------------------------

	#---------- Create prior directory -------------
	os.makedirs(dir_prior,exist_ok=True)
	#------------------------------------------------

	#--------- Initialize the inference module -------
	kal = Inference(dimension=dimension,
					dir_out=dir_prior,
					zero_points=zero_points,
					indep_measures=indep_measures,
					reference_system=reference_system,
					sampling_space=sampling_space,
					velocity_model=velocity_model)

	#-------- Load the data set --------------------
	# It will use the Gaia column names by default.
	kal.load_data(file_data,
					radec_precision_arcsec=radec_precision_arcsec)

	#------ Prepares the model -------------------
	kal.setup(prior=prior["type"],
			  parameters=prior["parameters"],
			  hyper_parameters=prior["hyper_parameters"],
			  parametrization=prior["parametrization"],
			  )
	#============ Sampling with HMC ======================================
	#------- Run the sampler ---------------------
	kal.run(sample_iters=sample_iters,
			tuning_iters=tuning_iters,
			target_accept=target_accept,
			chains=chains,
			cores=cores,
			init_iters=int(1e5),
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
	kal.save_statistics()

	kal.save_posterior_predictive()

	#------- Save the samples into HDF5 file --------------
	kal.save_samples()
#=======================================================================================
