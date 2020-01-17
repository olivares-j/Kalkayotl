'''
Copyright 2018 Javier Olivares Romero

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
#------------ LOAD LIBRARIES -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
import numpy as np
import pandas as pn

from  kalkayotl import Inference

generic_name = "EFF"
number_of_stars = 500

#============ Directories =============================
#-------Main directory ---------------
dir_main  = "/home/javier/Repositories/Kalkayotl/"
# dir_main  = os.getcwd() +"/"
#----------- Data --------------------
dir_data  = dir_main + "Data/Synthetic/"
dir_outs  = dir_main + "Outputs/Synthetic/"


#------- Create directories -------
os.makedirs(dir_outs,exist_ok=True)

#===================== Chains =================================
burning_iters   = 1000
sample_iters    = 2000   # Number of iterations for the MCMC 
#==============================================================

random_seeds = [1]

#------------------------- Case----------------------------
list_of_cases = [
{"name":generic_name,"location":100,"size":10,  "case_factor":1,"parametrization":"central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":200,"size":20,  "case_factor":1,"parametrization":"central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":300,"size":30,  "case_factor":1,"parametrization":"central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":400,"size":40,  "case_factor":1,"parametrization":"central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":500,"size":50,  "case_factor":2,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":600,"size":60,  "case_factor":2,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":700,"size":70,  "case_factor":2,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":800,"size":80,  "case_factor":2,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":900,"size":90,  "case_factor":2,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":1000,"size":100,"case_factor":2,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":2000,"size":200,"case_factor":3,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":3000,"size":300,"case_factor":3,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":4000,"size":400,"case_factor":4,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"name":generic_name,"location":5000,"size":500,"case_factor":4,"parametrization":"non-central", 'init':'advi+adapt_diag'},
]

list_of_prior = [
	# {"type":"EDSD",         "parameters":{"location":0.0,"scale":1350.0}, 
	# 						"hyper_alpha":None, 
	# 						"hyper_beta":None, 
	# 						"hyper_gamma":None,
	# 						"hyper_delta": None},

	# {"type":"Uniform",      "parameters":{"location":None,"scale":None},
	# 						"hyper_beta":[100],
	# 						"hyper_gamma":None, 
	# 						"hyper_delta":None},

	# {"type":"Gaussian",     "parameters":{"location":None,"scale":None},
	# 						"hyper_beta":[100],
	# 						"hyper_gamma":None,
	# 						"hyper_delta":None},

	{"type":"EFF",          "parameters":{"location":None,"scale":None,"gamma":3.0}, 
							"hyper_beta":[100],
							"hyper_gamma":[3.0,1.0],
							"hyper_delta":None},

	# {"type":"King",         "parameters":{"location":None,"scale":None,"rt":None},
	# 						"hyper_beta":[100],
	# 						"hyper_gamma":[10.0],
	# 						"hyper_delta":None},

	# {"type":"GMM",          "parameters":{"location":None,"scale":None,"weights":None},
	# 						"hyper_beta":[100],
	# 						"hyper_gamma":None,
	# 						"hyper_delta":np.array([5,5])},

	# {"type":"Cauchy",       "parameters":{"location":None,"scale":None},
	# 						"hyper_beta":[100],
	# 						"hyper_gamma":None,
	# 						"hyper_delta":None},
	]



indep_measures = [
			{"bool":False,"name":"corr", "target_accept":0.9},
			{"bool":True, "name":"indep","target_accept":0.9}
			]


#============================ Loop over cases ==================================
for seed in random_seeds:
	name = generic_name +"_"+str(number_of_stars)+"_"+str(seed)

	for case in list_of_cases:
		#------------ Local directories -----------------------------------------------------
		file_data = dir_data + name +"/" + case["name"] + "_" + str(case["location"]) + ".csv"
		dir_case  = dir_outs + name +"/" + case["name"] + "_" + str(case["location"]) +"/"
		#--------------------------------------------------------------------------------

		os.makedirs(dir_case,exist_ok=True)

		#======================= Inference and Analysis =====================================================
		#--------------------- Loop over prior types ------------------------------------

		for prior in list_of_prior:
			for indep in indep_measures:
				#----------- Output dir -------------------
				dir_prior = dir_case + prior["type"]
				dir_out   = dir_prior + "/" +indep["name"]

				os.makedirs(dir_prior,exist_ok=True)
				os.makedirs(dir_out,exist_ok=True)

				#--------- Run model -----------------------
				p1d = Inference(dimension=1,
								prior=prior["type"],
								parameters=prior["parameters"],
								hyper_alpha=[case["location"],case["size"]],
								hyper_beta=prior["hyper_beta"],
								hyper_gamma=prior["hyper_gamma"],
								hyper_delta=prior["hyper_delta"],
								dir_out=dir_out,
								transformation="pc",
								parametrization=case["parametrization"],
								zero_point=0.0,
								indep_measures=indep["bool"])
				p1d.load_data(file_data,id_name="ID")
				p1d.setup()
				p1d.run(sample_iters=sample_iters,
						burning_iters=burning_iters*case["case_factor"],
						target_accept=indep["target_accept"],
						init=case['init'],
						n_init=500000,
						chains=2,cores=2)

				#-------- Analyse chains --------------------------------
				p1d.load_trace(sample_iters=sample_iters)
				p1d.convergence()
				p1d.plot_chains(dir_out)
				p1d.save_statistics(statistic="mean",quantiles=[0.05,0.95])
		#=======================================================================================