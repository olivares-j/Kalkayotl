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

statistic = "map"


#---------------- MCMC parameters  --------------------
sample_iters    = 100   # Number of iterations for the MCMC 
burning_iters   = 1000


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
#----------- prior parameters ---------------------------------------------------------------
list_of_prior = [
    "Gaussian"
    # "GMM"
    ]
#------- For GMM prior this represents the hyper-parameter of the Dirichlet distribution
hyper_gamma = None#np.array([2,1]) # Two components
#--------- Cluster values ------------
fac      = 5
ra,dra   = 289.02,1.15
dec,ddec = -16.43,1.57
plx,dplx = 3.25,0.14

#------------------------ 1D ----------------------------
if dimension == 1:
    idx = 0
    zero_point = -0.029

    #----------- Parameters --------
    #Either non or fixed value in pc
    parameters = {"location":None,"scale":None}
    transformation = "pc"

    #------ hyper-parameters ------------------------------
    if transformation is "mas":
        hyper_alpha = [[plx-dplx,plx+dplx]]
        hyper_beta  = [fac*dplx]
    else:
        hyper_alpha = [[0,1000]]
        hyper_beta  = [100]

#---------------------- 3D ---------------------------------
elif dimension == 3:
    idx = 2
    zero_point = np.array([0,0,-0.029])

    #----------- Parameters --------
    #Either non or fixed value in pc
    parameters = {"location":None,"scale":None}
    # parameters = {"location":[ra,dec,plx],"scale":np.diag([dra,ddec,dplx])}
    transformation = "pc"

    #------ hyper-parameters ------------------------------
    if transformation is "mas":
        hyper_alpha = [[ra-dra,ra+dra],[dec-ddec,dec+ddec],[plx-dplx,plx+dplx]]
        hyper_beta  = [dra,ddec,dplx,1]
    elif transformation is "pc":
        hyper_alpha = [[90,100],[-280,-270],[-90,-80]]
        hyper_beta  = [100,100,100,1]

#--------------------- 5D ------------------------------------
elif dimension == 5:
    idx = 2
    zero_point = np.array([0,0,-0.000029,0.010,0.010])

#-------------------- 6D -------------------------------------
elif dimension == 6:
    idx = 2
    zero_point = np.array([0,0,-0.000029,0.010,0.010,0.0])

else:
    sys.exit("Dimension is not correct")
#------------------------------------------

#------- Create directories -------
if not os.path.isdir(dir_ana):
	os.mkdir(dir_ana)
if not os.path.isdir(dir_case):
    os.mkdir(dir_case)
if not os.path.isdir(dir_chains):
    os.mkdir(dir_chains)
if not os.path.isdir(dir_plots):
    os.mkdir(dir_plots)
#---------------------------------

#======================= Inference and Analysis =====================================================
id_name = "ID"

for prior in list_of_prior:
    #----------- Output dir -------------------
    dir_out = dir_chains + prior
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    dir_out = dir_out + "/" + str(dimension)+"D"
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    #--------- Run model -----------------------
    if not os.listdir(dir_out):
        p1d = Inference(dimension=dimension,
                        prior=prior,
                        parameters=parameters,
                        hyper_alpha=hyper_alpha,
                        hyper_beta=hyper_beta,
                        hyper_gamma=hyper_gamma,
                        transformation=transformation,
                        zero_point=zero_point)
        p1d.load_data(file_data,id_name=id_name)
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