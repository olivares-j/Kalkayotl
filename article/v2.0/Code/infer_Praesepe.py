from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
os.environ["MKL_NUM_THREADS"] = "1" # Avoids overlapping of processes
os.environ["OMP_NUM_THREADS"] = "1" # Avoids overlapping of processes
import numpy as np
import h5py

# from groups import *


# dir_kal  = "/home/jromero/Repos/Kalkayotl"
# dir_main = dir_kal + "/article/v2.0/Praesepe/"
dir_kal  = "/home/jolivares/Repos/Kalkayotl"
dir_main = "/home/jolivares/Projects/Kalkayotl/Praesepe/"

authors = ["Jadhav+2024_wtr"]#,"GG+2023_wtr","Hao+2022_wtr""GG+2023_core","GG+Lodieu"]

#----- Import the module -------------------------------
sys.path.append(dir_kal)
from kalkayotl.inference import Inference
#-------------------------------------------------------

#=============== Tuning knobs ============================
dimension = 6
chains    = 2
cores     = 2
init_iters    = int(3e5)
tuning_iters  = 3000
sample_iters  = 2000
target_accept = 0.65
init_refine   = True

sampling_space   = "physical"
indep_measures   = False
nuts_sampler     = "numpyro"
# nuts_sampler     = "pymc"

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
# 							"location":None,
# 							"scale":None,
# 							"eta":None,
# 							},
# 		"parametrization":"central"}

prior = {"type":"Gaussian",
		"parameters":{"location":None,"scale":None,"kappa":None,"omega":None},
		"hyper_parameters":{
							"location":None,
							"scale":None, 
							"eta":None,
							"kappa":None,
							"omega":None
							},
		"parametrization":"central"
		}

# prior = {"type":"FGMM",      
# 		"parameters":{"location":None,
# 					  "scale":None,
# 					  "weights":None,
# 					  "field_scale":[20.,20.,20.,5.,5.,5.]
# 					  },
# 		"hyper_parameters":{
# 							"location":None,
# 							"scale":None, 
# 							"weights":{"a":np.array([8,2])},
# 							"eta":None,
# 							},
# 		"parametrization":"central"}

#======================= Inference and Analysis =====================================================
for author in authors:
	dir_base = "{0}{1}/".format(dir_main,author)
	file_data = "{0}members.csv".format(dir_base)

	dir_prior = dir_base +  "{0}D_{1}_{2}_linear_1E+06".format(
							dimension,
							prior["type"],
							rs)

	#------- Creates directory if it does not exists -------
	os.makedirs(dir_base,exist_ok=True)
	os.makedirs(dir_prior,exist_ok=True)
	#-------------------------------------------------------

	kal = Inference(dimension=dimension,
					dir_out=dir_prior,
					zero_points=zero_points,
					indep_measures=indep_measures,
					reference_system=rs,
					sampling_space=sampling_space
					)

	kal.load_data(file_data)

	kal.setup(prior=prior["type"],
			  parameters=prior["parameters"],
			  hyper_parameters=prior["hyper_parameters"],
			  parameterization=prior["parametrization"])

	kal.run(sample_iters=sample_iters,
			tuning_iters=tuning_iters,
			target_accept=target_accept,
			chains=chains,
			cores=cores,
			init_iters=init_iters,
			init_refine=init_refine,
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
