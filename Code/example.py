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

#--------------- Inferer -------------------
from inference import Inference

#-------------- Chain analyser -------------
# from chain_analyser import Analysis


#----------- Dimension and Case ---------------------
dimension = 3
# If synthetic, comment the zero_point line in inference.
case      = "Rup147"
file_csv  = "Rup147.csv"

# case      = "Cluster_300"
# file_csv  = "Cluster_300_20_random.csv"

#------ Cluster mean and dispersion -----------------------
#--- R.A., Dec. parallax,proper motions and radial velocity
mu     = np.array([289.02,-16.43, 3.25,0.0,0.0,0.0])
sd     = np.array([1.15, 1.57, 0.14,1.0,1.0,1.0])
#---------------------------------------------------------

statistic = "map"

id_name = "ID"


#---------------- MCMC parameters  --------------------
sample_iters    = 500   # Number of iterations for the MCMC 
burning_iters   = 2000


#============ Directories =================
#-------Main directory ---------------
dir_main  = os.getcwd().replace("Code","")

#-------------------------------------

#----------- Data --------------------
dir_data  = dir_main + "Data/"
# dir_data  = dir_main + "Data/Synthetic/"
file_data = dir_data + case +"/"+ file_csv
#-------------------------------------

#--------- Chains and plots ----------
dir_ana    = dir_main + "Analysis/"
dir_case   = dir_ana  + case +"/"
# dir_case   = dir_ana  + "Synthetic/" + case +"/"
dir_chains = dir_case + "Chains/"
dir_plots  = dir_case + "Plots/"
#--------------------------------------

#================== Posterior =============================
#----------- Prior type -----------------------------
# If Gaussian then hyper_gamma must be None
# If GMM then hyper_delta must be set (see below).
list_of_prior = [
	# "Gaussian"
	"GMM"
	]
#----------------------------------------------------

#----------- Hyper Alpha -------------------------------------

all_hyp_alpha_mas = (np.vstack([mu - sd, mu + sd]).T).tolist()
#-------------------------------------------------------------

#------------- Hyper Beta --------------------
# Hyper-parameter controlling the distribution of standard 
# deviations. Corresponds to beta in HalfCauchy distribution
all_hyp_beta_mas  = (5.0*sd).tolist()
all_hyp_beta_pc   = np.repeat(10,6).tolist()
#-------------------------------------------------

#------------ Hyper Gamma ---------------------------------
# Hyper-parameter controlling the correlation distribution
# Only used in D >= 3. See Pymc3 LKJCorr function.
hyper_gamma = 1
#----------------------------------------------------------

#------------ Hyper delta -------------------------------
# Only for GMM prior. It represents the hyper-parameter 
# of the Dirichlet distribution.
# Its length indicate the number of components in the GMM.
hyper_delta = np.array([2,1])
#----------------------------------------------------------

#----------- Parameters -----------------------------------------------------
# Represent the mean and standard deviation of the
# Gaussian or Gaussians. If set to None these will be inferred by the model.
# If set, attention must be paid to their shape in accordance to the model
# dimension (i.e. if D=5 scale is a 5X5 matrix and location a 5-vector)
parameters = {"location":None,"scale":None}
#-----------------------------------------------------------------------------

# --------- Transformation------------------------------------
# Either "mas" or "pc" to indicate the observable or physical
# space in which the Parameters will be inferred
transformation = "pc"
#--------------------------------------------------------------

#--------- Zero point ------------------- ------------
# The zero point of the astrometry
all_zero_pint = np.array([0,0,-0.029,0.010,0.010,0.0])
#--------------------------------------------------------


#==========================================================

#========================== Models ==========================
#------------------------ 1D ----------------------------
if dimension == 1:
	zero_point = all_zero_pint[2]

	#------ hyper-parameters ------------------------------
	if transformation is "mas":
		hyper_alpha = [all_hyp_alpha_mas[2]]
		hyper_beta  = [all_hyp_beta_mas[2]]
	else:
		hyper_alpha = [[0,1000]]
		hyper_beta  = [all_hyp_beta_pc[2]]

#---------------------- 3D ---------------------------------
elif dimension == 3:
	zero_point = all_zero_pint[:3]

	#------ hyper-parameters ------------------------------
	if transformation is "mas":
		hyper_alpha = all_hyp_alpha_mas[:3]
		hyper_beta  = all_hyp_beta_mas[:3]
	elif transformation is "pc":
		hyper_alpha = [[90,100],[-280,-270],[-90,-80]]
		hyper_beta  = all_hyp_beta_pc[:3]

#--------------------- 5D ------------------------------------
elif dimension == 5:
	zero_point = all_zero_pint[:5]

	#------ hyper-parameters ------------------------------
	if transformation is "mas":
		hyper_alpha = all_hyp_alpha_mas[:5]
		hyper_beta  = all_hyp_beta_mas[:5]
	elif transformation is "pc":
		hyper_alpha = [[90,100],[-280,-270],[-90,-80]]
		hyper_beta  = all_hyp_beta_pc[:5]

#-------------------- 6D -------------------------------------
elif dimension == 6:
	zero_point = all_zero_point

	#------ hyper-parameters ------------------------------
	if transformation is "mas":
		hyper_alpha = all_hyp_alpha_mas
		hyper_beta  = all_hyp_beta_mas
	elif transformation is "pc":
		hyper_alpha = [[90,100],[-280,-270],[-90,-80]]
		hyper_beta  = all_hyp_beta_pc

else:
	sys.exit("Dimension is not correct")
#------------------------------------------
#============================================================================================

#======================= Inference and Analysis =====================================================

#------- Create directories -------
os.makedirs(dir_ana,exist_ok=True)
os.makedirs(dir_case,exist_ok=True)
os.makedirs(dir_chains,exist_ok=True)
os.makedirs(dir_plots,exist_ok=True)
#---------------------------------

#--------------------- Loop over prior types ------------------------------------

for prior in list_of_prior:
	#----------- Output dir -------------------
	dir_prior = dir_chains + prior
	dir_out = dir_prior + "/" + str(dimension)+"D"

	os.makedirs(dir_prior,exist_ok=True)
	os.makedirs(dir_out,exist_ok=True)

	#--------- Run model -----------------------
	p1d = Inference(dimension=dimension,
					prior=prior,
					parameters=parameters,
					hyper_alpha=hyper_alpha,
					hyper_beta=hyper_beta,
					hyper_gamma=hyper_gamma,
					hyper_delta=hyper_delta,
					transformation=transformation,
					zero_point=zero_point)
	p1d.load_data(file_data,id_name=id_name,nrows=2)
	p1d.setup()
	p1d.run(sample_iters=sample_iters,
		burning_iters=burning_iters,
		dir_chains=dir_out)
	# sys.exit()

	# #----------------- Analysis ---------------
	# a1d = Analysis(n_dim=dimension,
	#                 file_name=file_chains,
	#                 id_name=id_name,
	#                 dir_plots=dir_plots,
	#                 tol_convergence=tolerance,
	#                 statistic=statistic,
	#                 quantiles=[0.05,0.95],
	#                 # transformation=None,
	#                 names="2",
	#                 transformation="ICRS2GAL",
	#                 )
	# a1d.plot_chains()
	# # a1d.save_statistics(file_csv)
#=======================================================================================