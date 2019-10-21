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


from  Kalkayotl.inference import kalkayotl


#------------------------- Case----------------------------
list_of_cases = [
# {"name":"Gaussian","location":100,"size":50},
# {"name":"Gaussian","location":200,"size":50},
# {"name":"Gaussian","location":300,"size":50},
# {"name":"Gaussian","location":400,"size":50},
# {"name":"Gaussian","location":500,"size":50},
# {"name":"Gaussian","location":600,"size":50},
{"name":"Gaussian","location":700,"size":50},
{"name":"Gaussian","location":800,"size":50},
{"name":"Gaussian","location":900,"size":50},
{"name":"Gaussian","location":1000,"size":50},
{"name":"Gaussian","location":1500,"size":50},
# {"name":"Gaussian","location":2000,"size":50},
# {"name":"Gaussian","location":2500,"size":50},
# {"name":"Gaussian","location":3000,"size":50},
# {"name":"Gaussian","location":3500,"size":50},
# {"name":"Gaussian","location":4000,"size":50},
# {"name":"Gaussian","location":4500,"size":50},
# {"name":"Gaussian","location":5000,"size":50},
# {"name":"Gaussian","location":10000,"size":50},
]

list_of_prior = [
	# {"type":"EDSD",         "parameters":{"location":0.0,"scale":1350.0}, 
	# 						"hyper_beta":None, 
	# 						"hyper_gamma":None,
	# 						"hyper_delta": None},

	# {"type":"Uniform",      "parameters":{"location":None,"scale":None},
	# 						"hyper_beta":[100.],
	# 						"hyper_gamma":None, 
	# 						"hyper_delta": None},

	{"type":"Gaussian",     "parameters":{"location":None,"scale":None},
							"hyper_beta":[100],
							"hyper_gamma":None,
							"hyper_delta": None},

	# {"type":"Gaussian",     "parameters":{"location":None,"scale":None},
	# 						"hyper_beta":[1000],
	# 						"hyper_gamma":None,
	# 						"hyper_delta": None},

	# {"type":"GMM",          "parameters":{"location":None,"scale":None},
	# 						"hyper_beta":[100.], 
	# 						"hyper_gamma":None,
	# 						"hyper_delta": np.array([0.5,0.5])},

	# {"type":"Cauchy",       "parameters":{"location":None,"scale":None},
	# 						"hyper_beta":[100.], 
	# 						"hyper_gamma":None,
	# 						"hyper_delta": None},

	# {"type":"EFF",          "parameters":{"location":None,"scale":None},
	# 						"hyper_beta":[100.], 
	# 						"hyper_gamma":[2.0,1.0],
	# 						"hyper_delta": None},

	# {"type":"King",         "parameters":{"location":None,"scale":None},
	# 						"hyper_beta":[100.], 
	# 						"hyper_gamma":[20.0],
	# 						"hyper_delta": None},
	]



indep_measures = [
			{"bool":True, "name":"indep"},
			{"bool":False,"name":"corr"}
			]

#===================== Chains =================================
#---------------- MCMC parameters  --------------------
burning_iters   = 40000
sample_iters    = 10000   # Number of iterations for the MCMC 


#------- Statistic ----------------------
# Statistics to return after chain analysis 
statistic = "mode"
quantiles = [0.05,0.95]
id_name   = "ID"          # Identifier's name
# --------- Transformation------------------------------------
# Either "mas" or "pc" to indicate the observable or physical
# space in which the Parameters will be inferred
transformation = "pc"

#==============================================================


#============ Directories =============================
#-------Main directory ---------------
dir_main  = os.getcwd() +"/"
#----------- Data --------------------
dir_data  = dir_main + "Data/Synthetic"+"/Gaussian_20/"
dir_outs  = dir_main + "Outputs/Synthetic/"

#------- Create directories -------
os.makedirs(dir_outs,exist_ok=True)

#============================ Loop over cases ==================================
for case in list_of_cases:
	file_data = dir_data + case["name"]+ "_" + str(case["location"]) + ".csv"
	#-----------------------------------------------------------------------

	#------- Create directories -------
	dir_case   = dir_outs  + case["name"] + "_" + str(case["location"]) +"/"
	os.makedirs(dir_case,exist_ok=True)

	#======================= Inference and Analysis =====================================================
	#--------------------- Loop over prior types ------------------------------------

	for prior in list_of_prior:
		for indep in indep_measures:
			#----------- Output dir -------------------
			dir_prior = dir_case + prior["type"]
			dir_out   = dir_prior + "/" +indep["name"]#+"_beta_"+str(prior["hyper_beta"][0])

			os.makedirs(dir_prior,exist_ok=True)
			os.makedirs(dir_out,exist_ok=True)

			#--------- Run model -----------------------
			p1d = kalkayotl(dimension=1,
							prior=prior["type"],
							parameters=prior["parameters"],
							hyper_alpha=[case["location"],case["size"]],
							hyper_beta=prior["hyper_beta"],
							hyper_gamma=prior["hyper_gamma"],
							hyper_delta=prior["hyper_delta"],
							dir_out=dir_out,
							transformation=transformation,
							zero_point=0.0,
							indep_measures=indep["bool"])
			p1d.load_data(file_data,id_name=id_name)
			p1d.setup()
			p1d.run(sample_iters=sample_iters,
					burning_iters=burning_iters,
					target_accept=0.95
					)

			#-------- Analyse chains --------------------------------
			p1d.load_trace(burning_iters=burning_iters)
			p1d.convergence()
			coords = {"flavour_1d_source_dim_0" : range(5)}
			p1d.plot_chains(dir_out,coords=coords)
			p1d.save_statistics(dir_csv=dir_out,
								statistic=statistic)
	#=======================================================================================