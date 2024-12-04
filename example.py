'''
Copyright 2024 Javier Olivares Romero

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

#----- Import the module -------------------------------
dir_kalkayotl  = "/home/jolivares/Repos/Kalkayotl/" 
sys.path.append(dir_kalkayotl)
from kalkayotl.inference import Inference
#-------------------------------------------------------

#============ Directory and data ===========================================
#---- Diriectory for input and output -------------------------------------------
dir_base = "/home/jolivares/Kalkayotl/"
#--------------------------------------------------------------------------------

#----------- Input data --------------------------
file_data = dir_base + "example.csv"
#-------------------------------------------------

#---------- File with parameters that will be kept fixed (Optional) --------
#file_parameters = dir_base + "Cluster_statistics.csv"
#----------------------------------------------------------------------------
#============================================================================

#=============== Tuning knobs ============================
#------ Dimensionality of the model: 1, 3 or 6
dimension = 1
#----------------------------------------------

#----------------- Chains-----------------------------------------------------
# The number of parallel chains you want to run. Two are the minimum required
# to analyse convergence.
chains = 2

# Number of computer cores to be used. You can increase it to run faster.
# IMPORTANT. Depending on your computer configuration you may have different performances.
# I recommend to use 2 cores; this is one per chain.
cores  = 2

# tuning_iters is the number of iterations used to tune the sampler
# These will not be used for the statistics nor the plots. 
# If the sampler shows warnings you probably must increase this value.
tuning_iters = 1000

# After discarding the burning you will obtain sample_iters*chains samples
# from the posterior distribution. These are the ones used in the plots and to
# compute statistics.
sample_iters = 1000


#----- Target_accept-------
# This parameter controls the acceptance of the proposed steps in the Hamiltonian
# Monte Carlo sampler. Increasing it helps in the convergence of the sampler 
# but increases the computing time.
target_accept = 0.65
#---------------------------------------------------------------------------

#------------ Statistic -------------------------------------------------------
# mean, sd and high density interval will be computed for the posterior of each parameter.
# The outputs will be at Source_statistics.csv and Cluster_statistics.csv files
hdi_prob = 0.95 # Equivalent to 2sigma
#------------------------------------------------------------------------------

# --------- Sampling space -----------------------------------------------------------------
# In which space you want to sample: "physical" or "observed"?
# "observed" works only in the 1D case where the sampling can be done in the parallax space.
# IMPORTANT: The units of the parameters and hyper-parameters
# defined below must coincide with those of the chosen transformation.
sampling_space = "physical"
#--------------------------------------------------------------------------------------------

#--------- Zero point -----------------------
# A dcictionary with zerpoints
zero_points = {
"ra":0.,
"dec":0.,
"parallax":-0.017,# This is Brown+2020 value
"pmra":0.,
"pmdec":0.,
"radial_velocity":0.}  
#--------------------------------------------

#------- Independent measurements------------------------------------------------------------
# In the Gaia astrometric data the measurements of stars are correlated between sources.
# By default, Kalkayotl will not assume independence amongst sources.
# Set it to True if you want to assume independence, 
# and thus neglect the parallax and proper motions angular correlations. 
indep_measures = False
#------------------------------------------------------------------------------------------

#---------- NUTS Sampler ----------------------------------------------
# This is the type of sampler to use.
# Check PyMC documentation for valid samplers and their installation
# By default use the "pymc" sampler.
nuts_sampler = "numpyro"
#----------------------------------------------------------------------

#================== Only for 3D or 6D ==================================================================
#------------- Reference system -----------
# Coordinate system in which parameters will be inferred
# Either "ICRS" or "Galactic". Only for 3D and 6D versions.
reference_system = "Galactic"

#------ Parametrization -----------------
# The performance of the HMC sampler can be improved by non-central parametrizations.
# Kalkayotl comes with two options: central and non-central. While the former works better
# for nearby clusters (<500 pc) the latter does it for faraway clusters (>500 pc).
parameterization = "central"
#-----------------------------------------------------------------------------------------

#----------- Velocity model ---------------------------------------------------------------------
# Different types of velocity models are implemented:
# "joint": this is the most general which results in a joint model in position+velocity
# "constant": models the velocity as expanding or contracting field
# "linear": models the velocity field as a linear function of position.
# These options are chosen by specifying the parameters within the list of "parameters".
# For example, to use the joint model then "kappa" and "omega" must NOT be in the dictionary 
# of parameters. If only the constant model of velocity is wanted, then "kappa" must be included
# but NOT "omega". If the linear field is wanted, then "kappa" and "omega" must be included
# Note: use None in each parameter to infer its value. To keep it fixed, pass the desired value.
#----------------------------------------------------------------------------------------------
#=======================================================================================================

#========================= PRIORS ===========================================
priors = {
"FGMM":{"type":"FGMM",
		"parameters":{"location":None,
					  "scale":None,
					  "weights":None,
					  "field_scale":[20.,20.,20.,5.,5.,5.]
					  },
		"hyper_parameters":{
							"location":None,
							"scale":None, 
							"weights":{"a":np.array([8,2])},
							"eta":None,
							},
		},
"GMM":{"type":"GMM",
		"parameters":{"location":None,
					  "scale":None,
					  "weights":None
					  },
		"hyper_parameters":{
							"location":None,
							"scale":None, 
							"weights":{"n_components":2}, # You can also pass it as in the FGMM case.
							"eta":None
							},
		},
"Gaussian":{"type":"Gaussian",
		"parameters":{"location":None,"scale":None},
		"hyper_parameters":{
							"location":None,
							"scale":None, # For example, to pass the location of the prior do {"loc":[20.0,20.0,20.0,0.5,0.5,0.5]},
							"eta":None
							},
		},
"Gaussian_linear":{"type":"Gaussian",
		"parameters":{"location":None,"scale":None,"kappa":None,"omega":None},
		"hyper_parameters":{
							"location":None,
							"scale":None,
							"eta":None,
							"kappa":None,
							"omega":None
							},
		},
"StudentT":{"type":"StudentT",
		"parameters":{"location":None,"scale":None,"nu":None},
		"hyper_parameters":{
							"location":None,
							"scale":None,
							"nu":None,
							"eta":None,
							"nu":None,
							},
		},
}
#======================= Inference and Analysis =====================================================

prior = priors["Gaussian"]

#------ Output directories for each prior ------------------------
dir_out = dir_base +  "{0}D_{1}".format(dimension,prior["type"])
#-----------------------------------------------------------------

#---------- Create prior directory -------------
os.makedirs(dir_out,exist_ok=True)
#------------------------------------------------

#--------- Initialize the inference module -------
kal = Inference(dimension=dimension,
				dir_out=dir_out,
				zero_points=zero_points,
				indep_measures=indep_measures,
				reference_system=reference_system,
				sampling_space=sampling_space)
#---------------------------------------------------

#-------- Load the data set --------------------
# It will use the Gaia column names by default.
kal.load_data(file_data)
#------------------------------------------------

#------ Prepares the model ---------------------------
kal.setup(prior=prior["type"],
		  parameters=prior["parameters"],
		  hyper_parameters=prior["hyper_parameters"],
		  parameterization=parameterization)
#-----------------------------------------------------

#------- Run the sampler ---------------------
kal.run(
		tuning_iters=tuning_iters,
		sample_iters=sample_iters,
		target_accept=target_accept,
		chains=chains,
		cores=cores,
		init_iters=int(1e5),
		nuts_sampler=nuts_sampler,
		prior_predictive=True,
		prior_iters=chains*sample_iters,
		progressbar=True,)
#-------------------------------------

# -------- Load the chains --------------------------------
# This is useful if you have already computed the chains
# and want to re-analyse
kal.load_trace()
#------------------------------------------------------------

# ------- Convergence statistics ---
kal.convergence()
#------------------------------------

#-------- Plot the trace of the chains ---------------------------------------------------------
# If you provide the list of IDs (string list) it will plot the traces of the provided sources. 
# If IDs keyword is set to None (default), it only plots the population parameters.
kal.plot_chains()
#-----------------------------------------------------------------------------------------------

#--- Check Prior and Posterior ----
kal.plot_prior_check()
# Note: to plot the prior the prior_predictive=True in run()
#----------------------------------

#--- Plot model --------------------------------------------------------------------------------
kal.plot_model()
# Pass n_samples to draw the disered number of lines (must be <= than total number of samples)
# To control the plotting options see the function where you can pass names and colors.
# ----------------------------------------------------------------------------------------------

#----- Compute and save the posterior statistics ---------
kal.save_statistics(hdi_prob=hdi_prob)
#---------------------------------------------------

#------- Save the samples --------------
#if you need the samples for future use you can save them in an h5 file
# kal.save_samples()
#=======================================================================================
