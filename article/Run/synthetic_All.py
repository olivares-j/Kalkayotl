'''
Copyright 2019 Javier Olivares Romero

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
#------------ Load libraries -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
import numpy as np

#----- Import the module ----------
from kalkayotl import Inference


#============ Data and Directories =============================
#-------Main directory ---------------
dir_main  =  "/home/javier/Repositories/Kalkayotl/article/"
#-------------------------------------

#--------- Directory where chains and plots will be saved ----
dir_out    = dir_main + "Outputs/Comparison/"
#--------------------------------------

#----------- Data file --------------------
file_data = dir_main + "Synthetic/Gaussian_500_4/Gaussian_500.csv"
#-----------------------------------------

#------- Creates directory -------
os.makedirs(dir_out,exist_ok=True)
#---------------------------------
#==================================================


#=============== Tuning knobs ============================
# --------- Transformation------------------------------------
transformation = "pc"

#---------------
chains = 2
cores  = 2
burning_iters = 1000  
sample_iters = 2000

# Initialization mode
init_mode = 'advi+adapt_diag'

#---- Iterations to run the advi algorithm-----
init_iter = 500000 

#------------ Statistic ----------------------------------------------
statistic = "mean"
quantiles = [0.025,0.975]
#----------------------------------------------------------------------

#--------- Zero point -----------------------------------------------
zero_point = 0.0
#---------------------------------------------------------------------

#------ Parametrization -----------------
parametrization="central"
#==========================================================

#============= hyper-parameters ================================================
hyper_alpha = [500.,50.]
hyper_beta = [100.]


#========================= PRIORS ===========================================
# Uncomment those prior families that you are interested in using. 
list_of_prior = [
	# {"type":"EDSD",         "parameters":{"location":0.0,"scale":1350.0}, 
	# 						"hyper_alpha":None, 
	# 						"hyper_beta":None, 
	# 						"hyper_gamma":None,
	# 						"hyper_delta": None,
	# 						"burning_factor":1,
	# 						"target_accept":0.9},

	# {"type":"Uniform",      "parameters":{"location":None,"scale":None},
	# 						"hyper_alpha":hyper_alpha,
	# 						"hyper_beta":hyper_beta,
	# 						"hyper_gamma":None, 
	# 						"hyper_delta":None,
	# 						"burning_factor":1,
	# 						"target_accept":0.8},

	# {"type":"Gaussian",     "parameters":{"location":None,"scale":None},
	# 						"hyper_alpha":hyper_alpha,
	# 						"hyper_beta":hyper_beta,
	# 						"hyper_gamma":None,
	# 						"hyper_delta":None,
	# 						"burning_factor":1,
	# 						"target_accept":0.8},

	{"type":"King",         "parameters":{"location":None,"scale":None,"rt":None},
							"hyper_alpha":hyper_alpha, 
							"hyper_beta":hyper_beta, 
							"hyper_gamma":[10.0],
							"hyper_delta":None,
							"burning_factor":10,
							"target_accept":0.95},
	# # NOTE: the tidal radius and its parameters are scaled.

	
	# {"type":"EFF",          "parameters":{"location":None,"scale":None,"gamma":None},
	# 						"hyper_alpha":hyper_alpha,
	# 						"hyper_beta":hyper_beta, 
	# 						"hyper_gamma":[0.5],
	# 						"hyper_delta":None,
	# 						"burning_factor":10,
	# 						"target_accept":0.95},
	# NOTE: the mean of the Gamma parameter will be at 1.0 + hyper_gamma

	# {"type":"GMM",          "parameters":{"location":None,"scale":None,"weights":None},
	# 						"hyper_alpha":hyper_alpha, 
	# 						"hyper_beta":[50.0], 
	# 						"hyper_gamma":None,
	# 						"hyper_delta":np.array([5,5]),
	# 						"burning_factor":10,
	# 						"target_accept":0.95}
	# NOTE: If you face failures of the style zero derivative try reducing the hyper_beta value.
	]

indep_measures = [
			{"bool":False,"name":"corr"},
			# {"bool":True, "name":"indep"}
			]
#======================= Inference and Analysis =====================================================

#--------------------- Loop over prior types ------------------------------------
for prior in list_of_prior:
	#------ Output directories for each prior -------------------
	dir_prior = dir_out + prior["type"] + "/"

	#---------- Create prior directory -------------
	os.makedirs(dir_prior,exist_ok=True)
	#------------------------------------------------

	for indep in indep_measures:

		#----------- Output dir -------------------
		dir_case   = dir_prior + "/" +indep["name"]
		os.makedirs(dir_case,exist_ok=True)

		#--------- Initialize the inference module ----------------------------------------
		p1d = Inference(dimension=1,                       # For now it only works in 1D.
						prior=prior["type"],
						parameters=prior["parameters"],
						hyper_alpha=prior["hyper_alpha"],
						hyper_beta=prior["hyper_beta"],
						hyper_gamma=prior["hyper_gamma"],
						hyper_delta=prior["hyper_delta"],
						dir_out=dir_case,
						transformation=transformation,
						zero_point=zero_point,
						indep_measures=indep["bool"],
						parametrization=parametrization)
		#-------- Load the data set --------------------
		# It will use the Gaia column names by default.
		p1d.load_data(file_data,id_name="ID")

		#------ Prepares the model -------------------
		p1d.setup()

		#============ Sampling with HMC ======================================

		#------- Run the sampler ---------------------
		p1d.run(sample_iters=sample_iters,
				burning_iters=burning_iters*prior["burning_factor"],
				init=init_mode,
				n_init=init_iter,
				target_accept=prior["target_accept"],
				chains=chains,
				cores=cores)

		# -------- Load the chains --------------------------------
		p1d.load_trace(sample_iters=sample_iters)

		# ------- Re-analyse the convergence of the sampler---
		p1d.convergence()

		#-------- Plot the trace of the chains ------------------------------------
		p1d.plot_chains(dir_case)#,IDs=['4087735025198194176'])

		#----- Compute and save the posterior statistics ---------
		p1d.save_statistics(statistic=statistic,quantiles=quantiles)

#=======================================================================================