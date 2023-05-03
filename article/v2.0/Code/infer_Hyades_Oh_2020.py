from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
os.environ["MKL_NUM_THREADS"] = "1" # Avoids overlapping of processes
os.environ["OMP_NUM_THREADS"] = "1" # Avoids overlapping of processes
import numpy as np
import h5py

# dir_base = "/raid/jromero/Kalkayotl/Hyades/Oh_2020/Core/"
# dir_kalkayotl  = "/home/jromero/Repos/Kalkayotl/" 

dir_base = "/home/jolivares/Repos/Kalkayotl/article/v2.0/Hyades/Oh_2020/Core/"
dir_kalkayotl  = "/home/jolivares/Repos/Kalkayotl/" 

#----- Import the module -------------------------------
sys.path.append(dir_kalkayotl)
from kalkayotl.inference import Inference
#-------------------------------------------------------

#----------- Data file -----------------------------------------------------
file_data = dir_base + "members_core_GDR2.csv"
file_parameters = None
#----------------------------------------------------------------------------

#------- Creates directory if it does not exists -------
os.makedirs(dir_base,exist_ok=True)
#-------------------------------------------------------

#=============== Tuning knobs ============================
dimension = 6
chains    = 2
cores     = 2
tuning_iters  = 3000
sample_iters  = 2000
target_accept = 0.95

sampling_space   = "physical"
indep_measures   = False
velocity_model   = "joint"
nuts_sampler     = "numpyro"

zero_points = {
"ra":0.,
"dec":0.,
"parallax":-0.029, # Lindegren A&A 616, A2 (2018)
"pmra":0.,
"pmdec":0.,
"radial_velocity":0.}  
#--------------------------------

prior = {"type":"Gaussian",
		"parameters":{"location":None,"scale":None},
		"hyper_parameters":{
							# "alpha":{
							# 	"loc":[-43.37,0.40,-17.46,-41.73,-19.29,-1.06],
							# 	"scl":[5.,5.,5.,5.,5.,5.]
							# 	},
							"alpha":None,
							"beta":[10.,10.,10.,1.,1.,1.],
							"gamma":None,
							"delta":None,
							"eta":None
							},
		"parametrization":"central"}

# prior = {"type":"FGMM",      
# 		"parameters":{"location":None,
# 					  "scale":None,
# 					  "weights":None,
# 					  "field_scale":[20.,20.,20.,10.,10.,10.]
# 					  },
# 		"hyper_parameters":{
# 							"alpha":None,
# 							"beta":None, 
# 							"delta":np.repeat(1,2),
# 							"eta":None,
# 							"n_components":2
# 							},
# 		"parametrization":"central"}

#======================= Inference and Analysis =====================================================
for reference_system in ["ICRS","Galactic"]:
	dir_prior = dir_base +  "{0}D_{1}_{2}_{3}".format(
							dimension,
							prior["type"],
							reference_system,
							velocity_model)

	os.makedirs(dir_prior,exist_ok=True)

	kal = Inference(dimension=dimension,
					dir_out=dir_prior,
					zero_points=zero_points,
					indep_measures=indep_measures,
					reference_system=reference_system,
					sampling_space=sampling_space,
					velocity_model=velocity_model)

	kal.load_data(file_data,
					corr_func="Lindegren+2018",
					radec_precision_arcsec=5.0,)

	kal.setup(prior=prior["type"],
			  parameters=prior["parameters"],
			  hyper_parameters=prior["hyper_parameters"],
			  parametrization=prior["parametrization"])

	# kal.run(sample_iters=sample_iters,
	# 		tuning_iters=tuning_iters,
	# 		target_accept=target_accept,
	# 		chains=chains,
	# 		cores=cores,
	# 		init_iters=int(2e6),
	# 		init_refine=True,
	# 		nuts_sampler=nuts_sampler,
	# 		posterior_predictive=True,
	# 		prior_predictive=True)

	# kal.load_trace()
	# kal.convergence()
	# kal.plot_chains()
	# kal.plot_prior_check()
	# kal.plot_model()
	# kal.save_statistics()
	# kal.save_samples()
#=======================================================================================
