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

#===================================================================
dir_base = "/home/jolivares/Repos/Kalkayotl/article/v2.0/Synthetic/"
dimension = 6
chains = 2
cores  = 2
tuning_iters = 1000
sample_iters = 1000
target_accept = 0.95
hdi_prob = 0.95
sampling_space = "physical"
reference_system = "Galactic"
zero_points = {
"ra":0.,
"dec":0.,
"parallax":-0.017,# This is Brown+2020 value
"pmra":0.,
"pmdec":0.,
"radial_velocity":0.}  
indep_measures = False
velocity_model = "joint"
nuts_sampler = "numpyro"
list_of_n_stars   = [100,200]
list_of_distances = [1600.0]
#=====================================================================

#========================= Cases ===========================================
list_of_cases = [
	{"type":"Gaussian",
		"parameters":{"location":None,"scale":None},
		"hyper_parameters":{
							"alpha":None,
							"beta":None,
							"gamma":None,
							"delta":None,
							"eta":None
							},
		"parametrization":"central"},
	# {"type":"StudentT",
	# 	"parameters":{"location":None,"scale":None},
	# 	"hyper_parameters":{
	# 						"alpha":None,
	# 						"beta":None,
	# 						"gamma":None,
	# 						"delta":None,
	# 						"eta":None,
	# 						"nu":None,
	# 						},
	# 	"parametrization":"non-central"},
	# {"type":"GMM",     
	# 	"parameters":{"location":None,
	# 				  "scale":None,
	# 				  "weights":None},
	# 	"hyper_parameters":{
	# 						"alpha":None,
	# 						"beta":None, 
	# 						"gamma":None,
	# 						"delta":np.repeat(1,2),
	# 						"eta":None,
	# 						"n_components":2
	# 						},
	# 	"parametrization":"central"},
	# {"type":"CGMM",     
	# 	"parameters":{"location":None,
	# 				  "scale":None,
	# 				  "weights":None},
	# 	"hyper_parameters":{
	# 						"alpha":None,
	# 						"beta":None, 
	# 						"gamma":None,
	# 						"delta":np.repeat(1,2),
	# 						"eta":None,
	# 						"n_components":2
	# 						},
	# 	"parametrization":"central"},
	# {"type":"FGMM",      
	# 	"parameters":{"location":None,
	# 				  "scale":None,
	# 				  "weights":None,
	# 				  "field_scale":[50.,50.,50.,10.,10.,10.][:dimension]
	# 				  },
	# 	"hyper_parameters":{
	# 						"alpha":None,
	# 						"beta":None, 
	# 						"delta":np.repeat(1,2),
	# 						"eta":None,
	# 						"n_components":2
	# 						},
	# 	"parametrization":"central"}
	]
#===============================================================================

#--------------------- Loop over case types ------------------------------------
for case in list_of_cases:
	for distance in list_of_distances:
		for n_stars in list_of_n_stars:
			#------ Directory and data file -------------------
			dir_case = dir_base +  "{0}D_{1}_n{2}_d{3}_{4}".format(
				dimension,
				case["type"],
				int(n_stars),
				int(distance),
				case["parametrization"])
			
			file_data = dir_base + "{0}_n{1}_d{2}.csv".format(
				case["type"],
				n_stars,
				distance)

			os.makedirs(dir_case,exist_ok=True)

			p3d = Inference(dimension=dimension,
							dir_out=dir_case,
							zero_points=zero_points,
							indep_measures=indep_measures,
							reference_system=reference_system)

			p3d.load_data(file_data)
			p3d.setup(prior=case["type"],
					  parameters=case["parameters"],
					  hyper_parameters=case["hyper_parameters"],
					  parametrization=case["parametrization"],
					  sampling_space=sampling_space,
					  velocity_model=velocity_model)

			p3d.run(sample_iters=sample_iters,
					tuning_iters=tuning_iters,
					target_accept=target_accept,
					chains=chains,
					cores=cores,
					nuts_sampler=nuts_sampler,
					posterior_predictive=True,
					prior_predictive=True)
			p3d.load_trace()
			p3d.convergence()
			p3d.plot_chains()
			p3d.plot_prior_check()
			p3d.plot_model(n_samples=10)
			p3d.save_statistics(hdi_prob=hdi_prob)
			p3d.save_posterior_predictive()
			p3d.save_samples()
#=======================================================================================
