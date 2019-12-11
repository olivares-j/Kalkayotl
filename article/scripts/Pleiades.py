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
from kalkayotl import Inference

#------------------------- Case----------------------------
case      = "Pleiades"   
distance  = 135.
#-----------------------------------------------------------

#============ Directories =============================
#-------Main directory ---------------
dir_main  = os.getcwd() +"/"

#----------- Data --------------------
file_data = dir_main + "Data/" + case +".csv"
#-------------------------------------

#--------- Chains and plots ----------
dir_out    = dir_main + "Outputs/Real/"
dir_case   = dir_out  + case +"/"
#--------------------------------------
#==================================================

#===================== Chains =================================
#---------------- MCMC parameters  --------------------
burning_iters   = 2000
sample_iters    = 1000 
#==============================================================

#===================== Hyper-prior=========================================
hyper_alpha = [distance,0.1*distance] 
hyper_beta  = [100.] 

list_of_prior = [
	# {"type":"Uniform",      "parameters":{"location":None,"scale":None},
	# 						"hyper_gamma":None, 
	# 						"hyper_delta":None},

	# {"type":"Gaussian",     "parameters":{"location":None,"scale":None},
	# 						"hyper_gamma":None,
	# 						"hyper_delta":None},

	# {"type":"EFF",          "parameters":{"location":None,"scale":None,"gamma":None}, 
	# 						"hyper_gamma":[3.0,1.0],
	# 						"hyper_delta":None},


	{"type":"King",         "parameters":{"location":None,"scale":None,"rt":None},
							"hyper_gamma":[100.],
							"hyper_delta":None},

	# {"type":"GMM",          "parameters":{"location":None,"scale":None,"weights":None},
	# 						"hyper_gamma":None,
	# 						"hyper_delta":np.array([0.7,0.2,0.1])},

	# {"type":"Cauchy",       "parameters":{"location":None,"scale":None},
	# 						"hyper_gamma":None,
	# 						"hyper_delta":None},
	] 
#========================================================================

#======================= Inference and Analysis =====================================================
#------- Create directories -------
os.makedirs(dir_out,exist_ok=True)
os.makedirs(dir_case,exist_ok=True)
#---------------------------------

#--------------------- Loop over prior types ------------------------------------

for prior in list_of_prior:
	#----------- Output dir -------------------
	dir_prior = dir_case + prior["type"] + "/"
	os.makedirs(dir_prior,exist_ok=True)

	#--------- Run model -----------------------
	p1d = Inference(dimension=1,
					prior=prior["type"],
					parameters=prior["parameters"],
					hyper_alpha=hyper_alpha,
					hyper_beta=hyper_beta,
					hyper_gamma=prior["hyper_gamma"],
					hyper_delta=prior["hyper_delta"],
					dir_out=dir_prior,
					transformation="pc",
					parametrization='central',
					zero_point=-0.029,
					indep_measures=False)
	p1d.load_data(file_data,id_name="ID")
	p1d.setup()
	p1d.run(sample_iters=sample_iters,
			burning_iters=burning_iters,
			chains=2,cores=2)

	#-------- Analyse chains --------------------------------
	p1d.load_trace(sample_iters=sample_iters)
	p1d.convergence()
	p1d.plot_chains(dir_prior)
	p1d.save_statistics(statistic="mode",quantiles=[0.05,0.95]) 
	# p1d.evidence(N_samples=100,M_samples=1000,dlogz=1.0,nlive=100,file=dir_prior+"Cluster_Z.csv")
#=======================================================================================