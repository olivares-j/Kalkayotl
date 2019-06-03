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
from chain_analyser import Analysis


#----------- Dimension and Case ---------------------
dimension = 1
# If synthetic, comment the zero_point line in inference.
case      = "Cluster_300"
statistic = "map"
file_csv  = "Cluster_300_20_random.csv"

#---------------- MCMC parameters  --------------------
sample_iters    = 1000    # Number of iterations for the MCMC 
burning_iters   = 1000


#============ Directories =================
#-------Main directory ---------------
dir_main  = os.getcwd().replace("Code","")

#-------------------------------------

#----------- Data --------------------
dir_data  = dir_main + "Data/"
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
#------------------------ 1D ----------------------------
if dimension == 1:
    idx = 0
    zero_point = -0.000029

    #----------- prior parameters --------
    list_of_prior = [
    "Gaussian"
    # "Cauchy",
    # "Uniform"
    ]

#---------------------- 3D ---------------------------------
elif dimension == 3:
    idx = 2
    from posterior_3d import Posterior
    zero_point = np.array([0,0,-0.000029])

    #----------- prior parameters ---------------------------------------------------------------
    list_of_prior = [
    {"type":["Uniform","Uniform","EDSD"],     
    "location":[180,0,0.0],   "scale":[180,90,1350.0]},

    # {"type":["Uniform","Uniform","Uniform"],  
    # "location":[180,0,300.0], "scale":[180,90,50.0]},

    # {"type":["Uniform","Uniform","Gaussian"], 
    # "location":[180,0,300.0], "scale":[180,90,50.0]},

    # {"type":["Uniform","Uniform","Cauchy"],   
    # "location":[180,0,300.0], "scale":[180,90,50.0]}
    ]

#--------------------- 5D ------------------------------------
elif dimension == 5:
    idx = 2
    from posterior_5d import Posterior
    zero_point = np.array([0,0,-0.000029,0.010,0.010])

    #----------- prior parameters --------
    list_of_prior = [
    {"type":["Uniform","Uniform","EDSD","Uniform","Uniform"],
    "location":[180,0,0.0,0,0],   "scale":[180,90,1350.0,500,500]},

    {"type":["Uniform","Uniform","Uniform","Uniform","Uniform"],
    "location":[180,0,300.0,0,0], "scale":[180,90,50.0,500,500]},

    {"type":["Uniform","Uniform","Gaussian","Uniform","Uniform"],
    "location":[180,0,300.0,0,0], "scale":[180,90,50.0,500,500]},

    {"type":["Uniform","Uniform","Cauchy","Uniform","Uniform"],
    "location":[180,0,300.0,0,0], "scale":[180,90,50.0,500,500]}
    ]

#-------------------- 6D -------------------------------------
elif dimension == 6:
    idx = 2
    from posterior_6d import Posterior
    zero_point = np.array([0,0,-0.000029,0.010,0.010,0.0])

    #----------- prior parameters --------
    list_of_prior = [
    {"type":["Uniform","Uniform","EDSD","Uniform","Uniform","Uniform"],
    "location":[180,0,0.0,0,0,0],   "scale":[180,90,1350.0,500,500,100]},

    {"type":["Uniform","Uniform","Uniform","Uniform","Uniform","Uniform"],
    "location":[180,0,300.0,0,0,0], "scale":[180,90,50.0,500,500,100]},

    {"type":["Uniform","Uniform","Gaussian","Uniform","Uniform","Uniform"],
    "location":[180,0,300.0,0,0,0], "scale":[180,90,50.0,500,500,100]},

    {"type":["Uniform","Uniform","Cauchy","Uniform","Uniform","Uniform"],
    "location":[180,0,300.0,0,0,0], "scale":[180,90,50.0,500,500,100]}
    ]

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

    #--------- Run model -----------------------
    if not os.listdir(dir_out):
        p1d = Inference(dimension=dimension,
                        prior=prior,
                        zero_point=zero_point)
        p1d.load_data(file_data,id_name=id_name)
        p1d.setup()
        p1d.run(sample_iters=sample_iters,
            burning_iters=burning_iters,
            dir_chains=dir_out)
    sys.exit()

    #----------------- Analysis ---------------
    a1d = Analysis(n_dim=dimension,
                    file_name=file_chains,
                    id_name=id_name,
                    dir_plots=dir_plots,
                    tol_convergence=tolerance,
                    statistic=statistic,
                    quantiles=[0.05,0.95],
                    # transformation=None,
                    names="2",
                    transformation="ICRS2GAL",
                    )
    a1d.plot_chains()
    # a1d.save_statistics(file_csv)
#=======================================================================================