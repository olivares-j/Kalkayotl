#------------ Load libraries -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
os.environ["MKL_NUM_THREADS"] = "1" # Avoids overlapping of processes
os.environ["OMP_NUM_THREADS"] = "1" # Avoids overlapping of processes
import numpy as np
import pandas as pd
import h5py
import dill
import time

dill.load_session(str(sys.argv[1]))
print(list_of_n_stars)
print(list_of_distances)
print(list_of_seeds)
print(model)
print(velocity_model)
# sys.exit()

dir_kalkayotl = "/home/jromero/Repos/Kalkayotl/"

#----- Import the module -------------------------------
sys.path.append(dir_kalkayotl)
from kalkayotl.inference import Inference
#-------------------------------------------------------

#----------------- Knobs ------------------------------
dimension = 6
chains = 2
cores  = 2
tuning_iters = 3000
sample_iters = 2000
target_accept = 0.65
sampling_space = "physical"
reference_system = "Galactic"
zero_points = {
"ra":0.,
"dec":0.,
"parallax":0.0,# This is Brown+2020 value
"pmra":0.,
"pmdec":0.,
"radial_velocity":0.}  
indep_measures = False
nuts_sampler = "numpyro"
sky_error_factor=1e6
#--------------------------------------------------

dir_base = "/raid/jromero/Kalkayotl/Synthetic/{0}_{1}/".format(model,velocity_model)

#========================= Cases ===========================================
if model == "Gaussian":
	case = {
		"parameters":{"location":None,"scale":None},
		"hyper_parameters":{
							"alpha":None,
							"beta":None,
							"gamma":None,
							"delta":None,
							"eta":None
							},
		}
elif model == "StudentT":
	case = {
		"parameters":{"location":None,"scale":None},
		"hyper_parameters":{
							"alpha":None,
							"beta":None,
							"gamma":None,
							"delta":None,
							"eta":None,
							"nu":None,
							}
			}
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
else:
	sys.exit("Model not recognized!")
#===============================================================================

#--------------------- Loop over case types ------------------------------------
execution_times = []
for distance in list_of_distances:
	for n_stars in list_of_n_stars:
		for seed in list_of_seeds:
			if distance <= 500:
				parametrization = "central"
			else:
				parametrization = "non-central"
		
			name = "{0}D_{1}_n{2}_d{3}_s{4}_{5}_{6:1.0E}".format(
				dimension,
				model,
				int(n_stars),
				int(distance),
				seed,
				parametrization,
				sky_error_factor)
			print(20*"-"+"  "+name+"  "+20*"-")

			#------ Directory and data file -------------------
			dir_case = dir_base + name
			
			file_data = dir_base + "{0}_n{1}_d{2}_s{3}.csv".format(
				model,
				int(n_stars),
				int(distance),
				seed)

			if os.path.isfile(dir_case+"/Chains.nc"):
				continue

			os.makedirs(dir_case,exist_ok=True)

			try:
				t0 = time.time()
				kal = Inference(dimension=dimension,
								dir_out=dir_case,
								zero_points=zero_points,
								indep_measures=indep_measures,
								reference_system=reference_system,
								sampling_space=sampling_space,
								velocity_model=velocity_model)
				kal.load_data(file_data,sky_error_factor=sky_error_factor)
				kal.setup(prior=model,
						  parameters=case["parameters"],
						  hyper_parameters=case["hyper_parameters"],
						  parametrization=parametrization)

				kal.run(sample_iters=sample_iters,
						tuning_iters=tuning_iters,
						target_accept=target_accept,
						chains=chains,
						cores=cores,
						init_iters=int(1e6),
						init_refine=True,
						step_size=None,
						nuts_sampler=nuts_sampler,
						prior_predictive=False)
				kal.load_trace()
				kal.convergence()
				kal.plot_chains()
				kal.plot_prior_check()
				kal.plot_model()
				kal.save_statistics()
				kal.save_posterior_predictive()
				kal.save_samples()
				t1 = time.time()
				execution_times.append(t1-t0)
			except Exception as e:
				print(e)
				print(10*"*"+" ERROR "+10*"*")

#=======================================================================================
df_times = pd.DataFrame(data={"Time":execution_times})
df_times.to_csv("./times_{0}.csv".format(velocity_model))
