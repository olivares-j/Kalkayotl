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

name = "Uniform"
number_of_stars = [100,500,1000]

#============ Directories =============================
#-------Main directory ---------------
dir_main  = "/home/javier/Repositories/Kalkayotl/"
# dir_main  = os.getcwd() +"/"
#----------- Data --------------------
dir_data  = dir_main + "Data/Synthetic/"
dir_outs  = dir_main + "Outputs/Synthetic/"+name+"/"


#------- Create directories -------
os.makedirs(dir_outs,exist_ok=True)

#===================== Chains =================================
burning_iters   = 1000
sample_iters    = 2000   # Number of iterations for the MCMC 
#==============================================================

random_seeds = [1,2,3,4,5,6,7,8,9,10]

#------------------------- Case----------------------------
list_of_cases = [
{"location":100, "fraction":0.1, "case_factor":1, "parametrization":"central"},
{"location":200, "fraction":0.1, "case_factor":1, "parametrization":"central"},
{"location":300, "fraction":0.1, "case_factor":1, "parametrization":"central"},
{"location":400, "fraction":0.1, "case_factor":1, "parametrization":"central"},
{"location":500, "fraction":0.1, "case_factor":1, "parametrization":"non-central"},
{"location":600, "fraction":0.1, "case_factor":1, "parametrization":"non-central"},
{"location":700, "fraction":0.1, "case_factor":1, "parametrization":"non-central"},
{"location":800, "fraction":0.1, "case_factor":1, "parametrization":"non-central"},
{"location":900, "fraction":0.1, "case_factor":1, "parametrization":"non-central"},
{"location":1000,"fraction":0.1, "case_factor":1, "parametrization":"non-central"},
{"location":2000,"fraction":0.1, "case_factor":1, "parametrization":"non-central"},
{"location":3000,"fraction":0.1, "case_factor":1, "parametrization":"non-central"},
{"location":4000,"fraction":0.1, "case_factor":1, "parametrization":"non-central"},
{"location":5000,"fraction":0.1, "case_factor":1, "parametrization":"non-central"},
]

prior = {   
"parameters":{"location":None,"scale":None},
"hyper_beta":[100],
"hyper_gamma":None, 
"hyper_delta":None
}

indep_measures = [
			{"bool":False,"name":"corr", "target_accept":0.8},
			{"bool":True, "name":"indep","target_accept":0.8}
			]

#============================ Loop over cases ==================================
for number in number_of_stars:
	for seed in random_seeds:
		for case in list_of_cases:
			#------------ Local directories -----------------------------------------------------
			file_data = dir_data + name + "_" + str(number) + "_" + str(seed) +"/" + name + "_" + str(case["location"]) + ".csv"
			dir_case  = dir_outs + name + "_" + str(number) + "_" + str(case["location"]) + "_" + str(seed) +"/"
			#--------------------------------------------------------------------------------

			os.makedirs(dir_case,exist_ok=True)

			#======================= Inference and Analysis =====================================================
			for indep in indep_measures:
				#----------- Output dir -------------------
				dir_out   = dir_case + "/" +indep["name"]
				os.makedirs(dir_out,exist_ok=True)

				#--------- Run model -----------------------
				p1d = Inference(dimension=1,
								prior=name,
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
						init='advi+adapt_diag',
						n_init=500000,
						chains=2,cores=2)

				#-------- Analyse chains --------------------------------
				p1d.load_trace(sample_iters=sample_iters)
				p1d.convergence()
				p1d.plot_chains(dir_out)
				p1d.save_statistics(statistic="mean",quantiles=[0.025,0.975])
			#=======================================================================================