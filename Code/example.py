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
dimension = 5
case      = "NGC6774"
statistic = "map"


#---------------Posterior -----------------
if dimension == 1:
    from posterior_1d import Posterior
    zero_point = -0.000029
elif dimension == 3:
    from posterior_3d import Posterior
    zero_point = np.array([0,0,-0.000029])
elif dimension == 5:
    from posterior_5d import Posterior
    zero_point = np.array([0,0,-0.000029,0.010,0.010])
elif dimension == 6:
    from posterior_6d import Posterior
    zero_point = np.array([0,0,-0.000029,0.010,0.010,0.0])
else:
    sys.exit("Dimension is not correct")
#------------------------------------------


#---------------- MCMC parameters  --------------------
n_iter    = 10000         # Number of iterations for the MCMC 
n_walkers = 50           # Number of walkers
tolerance = 20

#----------- prior parameters --------
list_of_prior = [
# {"type":"Uniform",      "location":0.0,"scale":1000.0},
# {"type":"Uniform",      "location":0.0,"scale":1350.0},
# {"type":"Uniform",      "location":0.0,"scale":1500.0},
# # {"type":"Half-Gaussian","location":0.0,"scale":1000.0},
# # {"type":"Half-Gaussian","location":0.0,"scale":1350.0},
# {"type":"Half-Gaussian","location":0.0,"scale":1500.0}
# # {"type":"Half-Cauchy",  "location":0.0,"scale":1000.0},
# # {"type":"Half-Cauchy",  "location":0.0,"scale":1350.0},
# {"type":"Half-Cauchy",  "location":0.0,"scale":1500.0},
# # {"type":"EDSD",         "location":0.0,"scale":1000.0},
# # {"type":"EDSD",         "location":0.0,"scale":1350.0},
# {"type":"EDSD",         "location":0.0,"scale":1500.0}
# {"type":"Gaussian",     "location":136.0,"scale":20.0},
# {"type":"Gaussian",     "location":136.0,"scale":40.0},
# {"type":"Gaussian",     "location":300.0,"scale":80.0}
# {"type":"Cauchy",       "location":136.0,"scale":20.0},
# {"type":"Cauchy",       "location":136.0,"scale":40.0},
{"type":"Cauchy",       "location":300.0,"scale":80.0}
]

#============ Directories =================
#-------Main directory ---------------
dir_main  = os.getcwd()[:-4]
#-------------------------------------

#----------- Data --------------------
dir_data  = dir_main + "Data/"+case+"/"
file_data = dir_data + "Olivares+2019.csv"
#-------------------------------------

#--------- Chains and plots ----------
dir_ana    = dir_main + "Analysis/"
dir_case   = dir_ana  + case +"/"
dir_chains = dir_case + "Chains/"
dir_plots  = dir_case + "Plots/"
#--------------------------------------


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
    name_chains = "Chains_"+str(dimension)+"D_"+str(prior["type"])+"_loc="+str(int(prior["location"]))+"_scl="+str(int(prior["scale"]))+".h5"
    file_chains = dir_chains + name_chains
    file_csv    = file_chains.replace("h5","csv")

    if not os.path.isfile(file_chains):
        p1d = Inference(posterior=Posterior,
                        prior=prior["type"],
                        prior_loc=prior["location"],
                        prior_scale=prior["scale"],
                        n_walkers=n_walkers,
                        zero_point=zero_point)
        p1d.load_data(file_data,id_name=id_name)
        p1d.run(n_iter,file_chains=file_chains,tol_convergence=tolerance)

    #----------------- Analysis ---------------
    a1d = Analysis(n_dim=dimension,file_name=file_chains,
                    id_name=id_name,
                    dir_plots=dir_plots,
                    tol_convergence=tolerance,
                    statistic=statistic,
                    quantiles=[0.05,0.95],
                    transformation=None)
    # a1d.plot_chains()
    a1d.save_statistics(file_csv)
#=======================================================================================