from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
os.environ["MKL_NUM_THREADS"] = "1" # Avoids overlapping of processes
os.environ["OMP_NUM_THREADS"] = "1" # Avoids overlapping of processes
import numpy as np
import h5py

user = "jolivares"
authors = "Oh_2020"
case = "GDR3"

dir_kalkayotl  = "/home/{0}/Repos/Kalkayotl/".format(user) 

#----- Import the module -------------------------------
sys.path.append(dir_kalkayotl)
from kalkayotl.inference import Inference
#-------------------------------------------------------

#----------- Directories and files -------------------------------
dir_oc = "/home/{0}/Repos/Kalkayotl/article/v2.0/Hyades/".format(user)
dir_base = "{0}{1}/{2}/".format(dir_oc,authors,case)
file_data = "{0}members.csv".format(dir_base)
#---------------------------------------------------------

#------- Creates directory if it does not exists -------
os.makedirs(dir_base,exist_ok=True)
#-------------------------------------------------------

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
velocity_model   = "linear"
nuts_sampler     = "numpyro"

if case == "GDR3":
	zero_points = {
	"ra":0.,
	"dec":0.,
	"parallax":-0.017,
	"pmra":0.,
	"pmdec":0.,
	"radial_velocity":0.}
	correlation = "Lindegren+2020"

elif case == "GDR2":
	zero_points = {
	"ra":0.,
	"dec":0.,
	"parallax":-0.029, # Lindegren A&A 616, A2 (2018)
	"pmra":0.,
	"pmdec":0.,
	"radial_velocity":0.}
	correlation = "Lindegren+2018"

rss = ["Galactic","ICRS"]
#--------------------------------

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

# prior = {"type":"FGMM",      
# 		"parameters":{"location":None,
# 					  "scale":None,
# 					  "weights":None,
# 					  "field_scale":[5.,5.,5.,10,10,10]
# 					  },
# 		"hyper_parameters":{
# 							"alpha":None,
# 							"beta":None, 
# 							"delta":np.array([8,2]),
# 							"eta":None,
# 							"n_components":2
# 							},
# 		"parametrization":"central"}

#======================= Inference and Analysis =====================================================
for rs in rss:
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
					corr_func=correlation,
					sky_error_factor=sky_error_factor)

	kal.setup(prior=prior["type"],
			  parameters=prior["parameters"],
			  hyper_parameters=prior["hyper_parameters"],
			  parametrization=prior["parametrization"])

	# kal.run(sample_iters=sample_iters,
	# 		tuning_iters=tuning_iters,
	# 		target_accept=target_accept,
	# 		chains=chains,
	# 		cores=cores,
	# 		init_iters=int(3e5),
	# 		init_refine=True,
	# 		step_size=None,
	# 		nuts_sampler=nuts_sampler,
	# 		prior_predictive=False)

	kal.load_trace()
	# kal.convergence()
	# kal.plot_chains()
	# kal.plot_prior_check()
	# kal.plot_model()
	# kal.save_statistics(hdi_prob=0.682689492137)
	# kal.save_statistics(hdi_prob=0.954499736104)
	# kal.save_statistics(hdi_prob=0.997300203937)
	# kal.save_statistics(hdi_prob=0.999936657516)
	# kal.save_statistics(hdi_prob=0.999999426697)
	kal.save_statistics()
	# kal.save_samples()
#=======================================================================================
