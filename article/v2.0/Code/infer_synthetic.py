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
from functions import filter_members

from groups import *

case = "Gaussian_linear"

globals_pkl = dir_syn + case + "_100/globals.pkl"
# globals_pkl = str(sys.argv[1])

dill.load_session(globals_pkl)

#----- Import the module -------------------------------
sys.path.append(dir_kal)
from kalkayotl.inference import Inference
#-------------------------------------------------------

#----------------- Knobs ------------------------------
dimension = 6
chains = 2
cores  = 2
tuning_iters = 4000
sample_iters = 2000
init_iters   = int(5e5)
target_accept = 0.65
init_refine = False
sampling_space = "physical"
reference_system = "Galactic"
zero_points = {
"ra":0.,
"dec":0.,
"parallax":0.0, # This is Brown+2020 value
"pmra":0.,
"pmdec":0.,
"radial_velocity":0.}  
indep_measures = True
nuts_sampler = "numpyro"
# nuts_sampler = "pymc"
#--------------------------------------------------

prior = PRIOR[case]

#--------------------- Loop over distance n_stars and seeds ------------------------------------
print(dir_base)
for n_stars in [400]:#list_of_n_stars:
	for distance in [400]:#list_of_distances:
		for seed in [3]:#list_of_seeds:
			tmp = "n{0}_d{1}_s{2}/".format(int(n_stars),int(distance),seed)
			print(40*"-" +" " + tmp + " " +40*"-")
			dir_tmp = dir_base + tmp

			if distance <= 500 or prior["type"] == "GMM":
				parametrization = "central"
			else:
				parametrization = "non-central"

			# if os.path.isfile(dir_tmp+"/Chains.nc"):
			# 	continue

			#------ Directory and data file -------------------
			file_data = dir_tmp + "synthetic.csv"
			file_mem  = dir_tmp + "members.csv"
			file_time = dir_tmp + "time.txt"

			filter_members(pd.read_csv(file_data),
					file_mem,
					radial_velocity_name="radial_velocity",
					parallax_limits={"min":-np.inf,"max":np.inf})

			try:
				t0 = time.time()
				kal = Inference(dimension=dimension,
								dir_out=dir_tmp,
								zero_points=zero_points,
								indep_measures=indep_measures,
								reference_system=reference_system,
								sampling_space=sampling_space)
				kal.load_data(file_mem)
				kal.setup(prior=prior["type"],
						  parameters=prior["parameters"],
						  hyper_parameters=prior["hyper_parameters"],
						  parameterization=parametrization)

				kal.run(sample_iters=sample_iters,
						tuning_iters=tuning_iters,
						target_accept=target_accept,
						chains=chains,
						cores=cores,
						init_iters=init_iters,
						init_refine=init_refine,
						step_size=None,
						nuts_sampler=nuts_sampler,
						prior_predictive=True)
				
				kal.load_trace()
				kal.convergence()
				kal.plot_chains()
				kal.plot_prior_check()
				kal.plot_model()
				kal.save_statistics()#chain=[1])
				kal.save_posterior_predictive()
				# kal.save_samples()
				dt = time.time() - t0
				with open(file_time, "a") as f:
				    print("{}".format(dt), file=f)
				
			except Exception as e:
				print(e)
				print(10*"*"+" ERROR "+10*"*")

#=======================================================================================
