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

dir_base = "/home/jolivares/Repos/Kalkayotl/article/v2.0/Synthetic/"
file_data = dir_base + "Gaussian_n100_d100_s0.csv"

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
"parallax":0.,
"pmra":0.,
"pmdec":0.,
"radial_velocity":0.}  
indep_measures = False
velocity_model = "joint"
nuts_sampler = "numpyro"

list_of_prior = [
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

for prior in list_of_prior:
	#------ Output directories for each prior -------------------
	dir_prior = dir_base +  "{0}D_{1}_{2}".format(
		dimension,
		prior["type"],
		prior["parametrization"])
	os.makedirs(dir_prior,exist_ok=True)
	kal = Inference(dimension=dimension,
					dir_out=dir_prior,
					zero_points=zero_points,
					indep_measures=indep_measures,
					reference_system=reference_system)

	
	kal.load_data(file_data)
	kal.setup(prior=prior["type"],
			  parameters=prior["parameters"],
			  hyper_parameters=prior["hyper_parameters"],
			  parametrization=prior["parametrization"],
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
