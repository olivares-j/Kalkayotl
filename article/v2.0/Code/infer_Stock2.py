from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
os.environ["MKL_NUM_THREADS"] = "1" # Avoids overlapping of processes
os.environ["OMP_NUM_THREADS"] = "1" # Avoids overlapping of processes
import numpy as np
import h5py

user = "jromero"
authors = ["GG+2023","Tarricq+2022"]

dir_kalkayotl  = "/home/{0}/Repos/Kalkayotl/".format(user) 

#----- Import the module -------------------------------
sys.path.append(dir_kalkayotl)
from kalkayotl.inference import Inference
#-------------------------------------------------------

#----------- Directories and files -------------------------------
# dir_oc = "/home/{0}/Repos/Kalkayotl/article/v2.0/Rup161/".format(user)
dir_oc = "/raid/jromero/Kalkayotl/Stock2/"
#---------------------------------------------------------

#=============== Tuning knobs ============================
dimension = 6
chains    = 2
cores     = 2
tuning_iters  = 3000
sample_iters  = 2000
target_accept = 0.65
sky_error_factor = 1e6

sampling_space   = "physical"
indep_measures   = False
velocity_model   = "joint"
nuts_sampler     = "pymc"

zero_points = {
"ra":0.,
"dec":0.,
"parallax":-0.017,
"pmra":0.,
"pmdec":0.,
"radial_velocity":0.}

rs = "Galactic"
#--------------------------------

# prior = {"type":"Gaussian",
# 		"parameters":{"location":None,"scale":None},
# 		"hyper_parameters":{
# 							"alpha":None,
# 							"beta":None,
# 							"gamma":None,
# 							"delta":None,
# 							"eta":None
# 							},
# 		"parametrization":"central"}

prior = {"type":"FGMM",      
		"parameters":{"location":None,
					  "scale":None,
					  "weights":None,
					  "field_scale":[20.,20.,20.,5.,5.,5.]
					  },
		"hyper_parameters":{
							"alpha":None,
							"beta":None, 
							"delta":np.array([8,2]),
							"eta":None,
							"n_components":2
							},
		"parametrization":"central"}

# prior = {"type":"GMM",      
# 		"parameters":{"location":None,
# 					  "scale":None,
# 					  "weights":None
# 					  },
# 		"hyper_parameters":{
# 							"alpha":None,
# 							"beta":None, 
# 							"delta":np.array([9,1]),
# 							"eta":None,
# 							"n_components":2
# 							},
# 		"parametrization":"central"}

#======================= Inference and Analysis =====================================================
for author in authors:
	dir_base = "{0}{1}/".format(dir_oc,author)
	file_data = "{0}members.csv".format(dir_base)
	dir_prior = dir_base +  "{0}D_{1}_{2}_{3}_{4:1.0E}".format(
							dimension,
							prior["type"],
							rs,
							velocity_model,
							sky_error_factor)

	os.makedirs(dir_prior,exist_ok=True)

	kal = Inference(dimension=dimension,
					dir_out=dir_prior,
					zero_points=zero_points,
					indep_measures=indep_measures,
					reference_system=rs,
					sampling_space=sampling_space,
					velocity_model=velocity_model)

	kal.load_data(file_data,
					sky_error_factor=sky_error_factor)

	kal.setup(prior=prior["type"],
			  parameters=prior["parameters"],
			  hyper_parameters=prior["hyper_parameters"],
			  parametrization=prior["parametrization"])

	kal.run(sample_iters=sample_iters,
			tuning_iters=tuning_iters,
			target_accept=target_accept,
			chains=chains,
			cores=cores,
			init_iters=int(5e5),
			init_refine=False,
			step_size=None,
			nuts_sampler=nuts_sampler,
			prior_predictive=False)

	kal.load_trace()
	kal.convergence()
	kal.plot_chains()
	kal.plot_prior_check()
	kal.plot_model()
	kal.save_statistics()
	# kal.save_statistics(hdi_prob=0.682689492137)
	# kal.save_statistics(hdi_prob=0.954499736104)
	# kal.save_statistics(hdi_prob=0.997300203937)
	# kal.save_statistics(hdi_prob=0.999936657516)
	# kal.save_statistics(hdi_prob=0.999999426697)
	# kal.save_samples()
#=======================================================================================
