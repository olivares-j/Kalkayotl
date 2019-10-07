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
from Transformations import astrometryToPhaseSpace

#--------------- Inferer -------------------
from inference import Inference


#------------------------- Case----------------------------
list_of_cases = [
# {"name":"Ruprecht","location":100},
# {"name":"Ruprecht","location":300},
# {"name":"Ruprecht_147","location":500},
# {"name":"Ruprecht_147","location":700},
# {"name":"NGC_1647","location":600},
# {"name":"NGC_1647","location":800},
# {"name":"NGC_1647","location":1000},
# {"name":"NGC_1647","location":1200},
{"name":"M67","location":900},
# {"name":"M67","location":1000},
# {"name":"M67","location":1100},
# {"name":"Pleiades","location":400},
]
#------------ List of dimensions -----------------------
list_of_dimensions = [1]

indep_measures = [
			{"bool":True, "sufix":"_indep"},
			{"bool":False,"sufix":"_corr"}
			]

#===================== Chains =================================
#---------------- MCMC parameters  --------------------
burning_iters   = 10000
sample_iters    = 5000   # Number of iterations for the MCMC 


#------- Statistic ----------------------
# Statistics to return after chain analysis 
statistic = "mode"
quantiles = [0.05,0.95]
#==============================================================


#============ Directories =============================
#-------Main directory ---------------
dir_main  = os.getcwd().replace("Code","")
#----------- Data --------------------
dir_data  = dir_main + "Data/Synthetic/"
dir_outs  = dir_main + "Outputs/Synthetic/"

#------- Create directories -------
os.makedirs(dir_outs,exist_ok=True)

#============================ Loop over cases ==================================
for case in list_of_cases:
	# If synthetic, comment the zero_point line in inference.
	id_name   = "ID"          # Identifier's name
	#-----------------------------------------------------------


	#------ Cluster obs mean and dispersion -----------------------
	#--- R.A., Dec. parallax,proper motions and radial velocity
	# mu     = np.array([289.02,-16.43, 3.25,-0.98,-26.69,40.0])
	# sd     = np.array([1.15, 1.57, 0.14,0.64,0.72,10.0])
	# mu_pc    = np.array([96,-277,-86,7,-26,-48])
	# sd_pc    = np.array([3,8,6,4,14,4],dtype="float32")
	# #---------------------------------------------------------

	file_data = dir_data +  case["name"]+ "/" + case["name"]+ "_" + str(case["location"]) + ".csv"
	#-----------------------------------------------------------------------

	#------- Create directories -------
	dir_case   = dir_outs  + case["name"] + "_" + str(case["location"]) +"/"
	os.makedirs(dir_case,exist_ok=True)
	#---------------------------------
	#==================================================

	#================== Posterior =============================
	#----------- Prior type and parameters-----------------------------

	#----------- Parameters -----------------------------------------------------
	# Represent the mean and standard deviation of the
	# Gaussian or Gaussians. If set to None these will be inferred by the model.
	# If set, attention must be paid to their shape in accordance to the model
	# dimension (i.e. if D=5 scale is a 5X5 matrix and location a 5-vector)
	# parameters = {"location":0.0,"scale":1500,"corr":False}
	#-----------------------------------------------------------------------------

	#------------ Hyper delta -------------------------------
	# Only for GMM prior. It represents the hyper-parameter 
	# of the Dirichlet distribution.
	# Its length indicate the number of components in the GMM.
	#----------------------------------------------------------

	list_of_prior = [
		# {"type":"EDSD",          "parameters":{"location":0.0,"scale":1350.0, "corr":False}, "hyper_delta": None},
		# {"type":"Uniform",       "parameters":{"location":None,"scale":None,  "corr":False}, "hyper_delta": None},
		# {"type":"Cauchy",        "parameters":{"location":None,"scale":None,  "corr":False}, "hyper_delta": None},
		{"type":"Gaussian",      "parameters":{"location":None,"scale":None,  "corr":False}, "hyper_delta": None},
		# {"type":"GMM",           "parameters":{"location":None,"scale":None,  "corr":False}, "hyper_delta": np.array([5,5])}
		]

	#----------------------------------------------------

	#----------- Hyper Alpha -------------------------------------
	# Hyper parameter describing the range where the mean values
	# will be searched for.
	# hyp_alpha_mas = np.vstack([mu - sd, mu + sd])
	# hyp_alpha_pc  = np.sort(astrometryToPhaseSpace(hyp_alpha_mas).T)
	# hyp_alpha_pc  = np.vstack([mu_pc - sd_pc, mu_pc + sd_pc]).tolist()
	# hyp_alpha_mas = hyp_alpha_mas.T.tolist()
	#-------------------------------------------------------------

	#------------- Hyper Beta --------------------
	# Hyper-parameter controlling the distribution of standard 
	# deviations. Corresponds to beta in HalfCauchy distribution
	# hyp_beta_mas  = (5.0*sd).tolist()
	# hyp_beta_pc   = np.repeat(1000,6).tolist()
	#-------------------------------------------------


	# --------- Transformation------------------------------------
	# Either "mas" or "pc" to indicate the observable or physical
	# space in which the Parameters will be inferred
	transformation = "pc"

	# if transformation is "mas":
	# 	hyp_alpha = hyp_alpha_mas
	# 	hyp_beta  = hyp_beta_mas
	# elif transformation is "pc":
	# 	hyp_alpha = hyp_alpha_pc
	# 	hyp_beta  = hyp_beta_pc
	#--------------------------------------------------------------

	#--------- Zero point ------------------- ------------
	# The zero point of the astrometry
	zero_point = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
	# zero_point = np.array([0,0,-0.029,0.010,0.010,0.0])
	#--------------------------------------------------------

	# #------- Independent measurements--------
	# # In Gaia real data the measurements of stars are correlated,
	# # thus indep_measures must be set to False (default)
	# indep_measures = False
	#==========================================================


	#--- Loop over dimension---------------------
	for dimension in list_of_dimensions:

		#------------ Hyper Gamma ---------------------------------
		# Hyper-parameter controlling the correlation distribution
		# Only used in D >= 3. See Pymc3 LKJCorr function.
		if dimension == 1:
			hyper_gamma = None
		else:
			hyper_gamma = 1
		#----------------------------------------------------------

		#========================== Models ==========================
		#------------------------ 1D ----------------------------
		if dimension == 1:
			zero_point  = zero_point[2]
			hyper_alpha = [[case["location"]-50,case["location"]+50]]
			hyper_beta  = [10.0]
			
		#---------------------- 3D ---------------------------------
		elif dimension == 3:
			zero_point  = zero_point[:3]
			hyper_alpha = hyp_alpha[:3]
			hyper_beta  = hyp_beta[:3]


		#--------------------- 5D ------------------------------------
		elif dimension == 5:
			zero_point  = zero_point[:5]
			hyper_alpha = hyp_alpha
			hyper_beta  = hyp_beta

		#-------------------- 6D -------------------------------------
		elif dimension == 6:
			zero_point  = zero_point
			hyper_alpha = hyp_alpha
			hyper_beta  = hyp_beta

		else:
			sys.exit("Dimension is not correct")
		#------------------------------------------
		#============================================================================================

		#======================= Inference and Analysis =====================================================
		#--------------------- Loop over prior types ------------------------------------

		for prior in list_of_prior:
			for indep in indep_measures:
				#----------- Output dir -------------------
				dir_prior = dir_case + prior["type"]
				dir_out   = dir_prior + "/" + str(dimension)+"D"+indep["sufix"]

				os.makedirs(dir_prior,exist_ok=True)
				os.makedirs(dir_out,exist_ok=True)

				#--------- Run model -----------------------
				p1d = Inference(dimension=dimension,
								prior=prior["type"],
								parameters=prior["parameters"],
								hyper_alpha=hyper_alpha,
								hyper_beta=hyper_beta,
								hyper_gamma=hyper_gamma,
								hyper_delta=prior["hyper_delta"],
								dir_out=dir_out,
								transformation=transformation,
								zero_point=zero_point,
								indep_measures=indep["bool"])
				p1d.load_data(file_data,id_name=id_name)
				p1d.setup()
				p1d.run(sample_iters=sample_iters,
						burning_iters=burning_iters,
						# target_accept=0.95,
						# step_scale=0.1
						)

				#-------- Analyse chains --------------------------------
				p1d.load_trace(burning_iters=burning_iters)
				p1d.convergence()
				coords = {"flavour_"+str(dimension)+"d_source_dim_0" : range(5)}
				p1d.plot_chains(dir_out,coords=coords)
				p1d.save_statistics(dir_csv=dir_out,
									statistic=statistic,
									quantiles=quantiles)
		#=======================================================================================