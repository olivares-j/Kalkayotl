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
import shutil
import numpy as np
import pandas as pn

from  Kalkayotl.inference import kalkayotl


#------------------------- Case----------------------------
# If synthetic, comment the zero_point line in inference.
case      = "NGC_6791"    # Case name
file_csv  = "NGC_6791.csv" # Data file
dst       = 4530.
id_name   = "ID"          # Identifier's name
#-----------------------------------------------------------

list_of_prior = [
	# {"type":"EDSD",         "parameters":{"location":0.0,"scale":1350.0}, 
	# 						"hyper_alpha":None, 
	# 						"hyper_beta":None, 
	# 						"hyper_gamma":None,
	# 						"hyper_delta": None},

	{"type":"Uniform",      "parameters":{"location":None,"scale":None},
							"hyper_alpha":[[dst,50.]],
							"hyper_beta":[100.],
							"hyper_gamma":None, 
							"hyper_delta":None},

	{"type":"Gaussian",     "parameters":{"location":None,"scale":None},
							"hyper_alpha":[[dst,50.]], 
							"hyper_beta":[100.],
							"hyper_gamma":None,
							"hyper_delta":None},

	{"type":"EFF",          "parameters":{"location":None,"scale":None},
							"hyper_alpha":[[dst,50.]], 
							"hyper_beta":[100.], 
							"hyper_gamma":[2.0,1.0],
							"hyper_delta":None},

	{"type":"GMM",          "parameters":{"location":None,"scale":None},
							"hyper_alpha":[[dst,50.]], 
							"hyper_beta":[100.], 
							"hyper_gamma":None,
							"hyper_delta":np.array([0.9,0.1])},

	{"type":"King",         "parameters":{"location":None,"scale":None},
							"hyper_alpha":[[dst,50.]], 
							"hyper_beta":[100.], 
							"hyper_gamma":[20.],
							"hyper_delta":None},

	{"type":"Cauchy",       "parameters":{"location":None,"scale":None},
							"hyper_alpha":[[dst,50.]], 
							"hyper_beta":[100.], 
							"hyper_gamma":None,
							"hyper_delta":None},
	]

#===================== Chains =================================
#---------------- MCMC parameters  --------------------
burning_iters   = 20000
sample_iters    = 10000   # Number of iterations for the MCMC 


#------- Statistic ----------------------
# Statistics to return after chain analysis 
statistic = "mode"
quantiles = [0.05,0.95]
#==============================================================


#============ Directories =============================
#-------Main directory ---------------
dir_main  = os.getcwd().replace("Code","")

#-------------------------------------

#----------- Data --------------------
file_data = dir_main + "Data/" + file_csv
#-------------------------------------

#--------- Chains and plots ----------
dir_out    = dir_main + "Outputs/"
dir_case   = dir_out  + case +"/"
#--------------------------------------
#==================================================


# --------- Transformation------------------------------------
# Either "mas" or "pc" to indicate the observable or physical
# space in which the Parameters will be inferred
transformation = "pc"


#--------- Zero point ------------------- ------------
# The zero point of the parallax
zero_point = -0.029
#--------------------------------------------------------

#------- Independent measurements--------
# In Gaia real data the measurements of stars are correlated,
# thus indep_measures must be set to False (default)
indep_measures = False
#==========================================================

#======================= Inference and Analysis =====================================================
#------- Create directories -------
os.makedirs(dir_out,exist_ok=True)
os.makedirs(dir_case,exist_ok=True)
#---------------------------------

#--------------------- Loop over prior types ------------------------------------

for prior in list_of_prior:
	#----------- Output dir -------------------
	dir_prior = dir_case + prior["type"]
	dir_out   = dir_prior + "/" 
	file_Z    = dir_out   + "Cluster_Z.csv"

	os.makedirs(dir_prior,exist_ok=True)

	# if os.path.exists(dir_out):
	# 	shutil.rmtree(dir_out)
	
	os.makedirs(dir_out,exist_ok=True)

	#--------- Run model -----------------------
	p1d = kalkayotl(dimension=1,
					prior=prior["type"],
					parameters=prior["parameters"],
					hyper_alpha=prior["hyper_alpha"],
					hyper_beta=prior["hyper_beta"],
					hyper_gamma=prior["hyper_gamma"],
					hyper_delta=prior["hyper_delta"],
					dir_out=dir_out,
					transformation=transformation,
					zero_point=zero_point,
					indep_measures=indep_measures,
					quantiles=quantiles)
	p1d.load_data(file_data,id_name=id_name)
	p1d.setup()
	p1d.evidence(N_samples=100,M_samples=1000,dlogz=1.0,nlive=100,file=file_Z,plot=True)
	
	p1d.run(sample_iters=sample_iters,
			burning_iters=burning_iters,
			# target_accept=0.95,
			)

	#-------- Analyse chains --------------------------------
	p1d.load_trace(burning_iters=burning_iters)
	p1d.convergence()
	coords = {"flavour_1d_source_dim_0" : range(5)}
	p1d.plot_chains(dir_out,coords=coords)
	p1d.save_statistics(dir_csv=dir_out,
						statistic=statistic)
#=======================================================================================