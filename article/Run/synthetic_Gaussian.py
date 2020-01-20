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

generic_name = "Gaussian"
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

random_seeds = [1]#,2,3,4,5,6,7,8,9,10]

#------------------------- Case----------------------------
list_of_cases = [
# {"location": 100,"fraction":0.1,"case_factor":1,"parametrization":"central",     'init':'advi+adapt_diag'},
# {"location": 200,"fraction":0.1,"case_factor":1,"parametrization":"central",     'init':'advi+adapt_diag'},
# {"location": 300,"fraction":0.1,"case_factor":1,"parametrization":"central",     'init':'advi+adapt_diag'},
# {"location": 400,"fraction":0.1,"case_factor":1,"parametrization":"central",     'init':'advi+adapt_diag'},
# {"location": 500,"fraction":0.1,"case_factor":1,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"location": 600,"fraction":0.1,"case_factor":1,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"location": 700,"fraction":0.1,"case_factor":1,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"location": 800,"fraction":0.1,"case_factor":1,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"location": 900,"fraction":0.1,"case_factor":1,"parametrization":"non-central", 'init':'advi+adapt_diag'},
{"location":1000,"fraction":0.1,"case_factor":1,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"location":2000,"fraction":0.1,"case_factor":1,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"location":3000,"fraction":0.1,"case_factor":1,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"location":4000,"fraction":0.1,"case_factor":1,"parametrization":"non-central", 'init':'advi+adapt_diag'},
# {"location":5000,"fraction":0.1,"case_factor":1,"parametrization":"non-central", 'init':'advi+adapt_diag'},
]

list_of_prior = [
	{"type":"EDSD",         "parameters":{"location":0.0,"scale":1350.0}, 
							"hyper_beta":None, 
							"hyper_gamma":None,
							"hyper_delta": None},

	# {"type":"Uniform",      "parameters":{"location":None,"scale":None},
	# 						"hyper_beta":[100],
	# 						"hyper_gamma":None, 
	# 						"hyper_delta":None},

	# {"type":"Gaussian",     "parameters":{"location":None,"scale":None},
	# 						"hyper_beta":[100],
	# 						"hyper_gamma":None,
	# 						"hyper_delta":None},

	# {"type":"EFF",          "parameters":{"location":None,"scale":None,"gamma":None}, 
	# 						"hyper_beta":[100],
	# 						"hyper_gamma":[3.0,1.0],
	# 						"hyper_delta":None},

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
			# {"bool":False,"name":"minus","target_accept":0.9,"value":-0.1},
			{"bool":False,"name":"corr", "target_accept":0.9,"value":0.0},
			# {"bool":False,"name":"plus", "target_accept":0.9,"value":0.1},
			{"bool":True, "name":"indep","target_accept":0.9,"value":0.0}
			]


#============================ Loop over cases ==================================
for seed in random_seeds:
	name = generic_name +"_"+str(number_of_stars)+"_"+str(seed)

	for case in list_of_cases:
		#------------ Local directories -----------------------------------------------------
		file_data = dir_data + name +"/" + generic_name + "_" + str(case["location"]) + ".csv"
		dir_case  = dir_outs + name +"/" + generic_name + "_" + str(case["location"]) +"/"
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
								hyper_alpha=[case["location"],case["fraction"]*case["location"]],
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