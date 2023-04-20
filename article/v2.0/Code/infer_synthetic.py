#------------ Load libraries -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
os.environ["MKL_NUM_THREADS"] = "1" # Avoids overlapping of processes
os.environ["OMP_NUM_THREADS"] = "1" # Avoids overlapping of processes
import numpy as np
import h5py
import dill

dill.load_session(str(sys.argv[1]))

dir_kalkayotl = "/home/jolivares/Repos/Kalkayotl/"
dir_base = "/home/jolivares/Repos/Kalkayotl/article/v2.0/Synthetic/"

#----- Import the module -------------------------------
sys.path.append(dir_kalkayotl)
from kalkayotl.inference import Inference
#-------------------------------------------------------

#----------------- Knobs ------------------------------
dimension = 6
chains = 2
cores  = 2
tuning_iters = 1000
sample_iters = 1000
target_accept = 0.95
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
#--------------------------------------------------

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
for distance in list_of_distances:
	for n_stars in list_of_n_stars:
		for seed in list_of_seeds:
		#------ Directory and data file -------------------
		dir_case = dir_base +  "{0}D_{1}_n{2}_d{3}_s{4}_{5}".format(
			dimension,
			case["type"],
			int(n_stars),
			int(distance),
			seed,
			case["parametrization"])
		
		file_data = dir_base + "{0}_n{1}_d{2}_s{3}.csv".format(
			case["type"],
			int(n_stars),
			int(distance),
			seed)

		os.makedirs(dir_case,exist_ok=True)

		kal = Inference(dimension=dimension,
						dir_out=dir_case,
						zero_points=zero_points,
						indep_measures=indep_measures,
						reference_system=reference_system)

		kal.load_data(file_data)
		kal.setup(prior=case["type"],
				  parameters=case["parameters"],
				  hyper_parameters=case["hyper_parameters"],
				  parametrization=case["parametrization"],
				  sampling_space=sampling_space,
				  velocity_model=velocity_model)

		kal.run(sample_iters=sample_iters,
				tuning_iters=tuning_iters,
				target_accept=target_accept,
				chains=chains,
				cores=cores,
				nuts_sampler=nuts_sampler,
				posterior_predictive=True,
				prior_predictive=True)
		kal.load_trace()
		kal.convergence()
		kal.plot_chains()
		kal.plot_prior_check()
		kal.plot_model()
		kal.save_statistics()
		kal.save_posterior_predictive()
		kal.save_samples()
#=======================================================================================
