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

#--------------- Posteriors and inferer -------------------
from inference import Inference
from posterior_5d import Posterior as Posterior_5d

#-------------- Chain analyser -------------
from chain_analyser import Analysis

#---------------- MCMC parameters  --------------------
n_iter    = 3000         # Number of iterations for the MCMC 
n_walkers = 30           # Number of walkers
tolerance = 10

#----------- prior parameters --------
list_of_prior = [
# {"type":"Uniform",      "location":0.0,"scale":1000.0},
# {"type":"Uniform",      "location":0.0,"scale":1350.0},
{"type":"Uniform",      "location":0.0,"scale":1500.0}
# {"type":"Half-Gaussian","location":0.0,"scale":1000.0},
# {"type":"Half-Gaussian","location":0.0,"scale":1350.0},
# {"type":"Half-Gaussian","location":0.0,"scale":1500.0},
# {"type":"Half-Cauchy",  "location":0.0,"scale":1000.0},
# {"type":"Half-Cauchy",  "location":0.0,"scale":1350.0},
# {"type":"Half-Cauchy",  "location":0.0,"scale":1500.0},
# {"type":"EDSD",         "location":0.0,"scale":1000.0},
# {"type":"EDSD",         "location":0.0,"scale":1350.0}
# {"type":"EDSD",         "location":0.0,"scale":1500.0},
# {"type":"Gaussian",     "location":300.0,"scale":20.0},
# {"type":"Gaussian",     "location":300.0,"scale":60.0},
# {"type":"Gaussian",     "location":300.0,"scale":100.0},
# {"type":"Cauchy",       "location":300.0,"scale":20.0},
# {"type":"Cauchy",       "location":300.0,"scale":60.0},
# {"type":"Cauchy",       "location":300.0,"scale":100.0}
]

#============ Directories and data =================

dir_main  = os.getcwd()[:-4]
dir_data  = dir_main + "Data/"
dir_expl  = dir_main + "Analysis/"
case      = "Ruprecht_147"
statistic = "map"
dir_chains= dir_expl + "Chains/"+case+"/"
dir_plots = dir_expl + "Plots/"+case+"/"
file_data = dir_data + case+".csv"

#------- Create directories -------
if not os.path.isdir(dir_expl):
	os.mkdir(dir_expl)
if not os.path.isdir(dir_chains):
    os.mkdir(dir_chains)
if not os.path.isdir(dir_plots):
    os.mkdir(dir_plots)
#==================================================

#======================= 5D Version =====================================================
list_observables = ["ID_member","ra","dec","parallax","pmra","pmdec",
                    "ra_error","dec_error","parallax_error","pmra_error","pmdec_error",
                "ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr",
                "dec_parallax_corr","dec_pmra_corr","dec_pmdec_corr",
                "parallax_pmra_corr","parallax_pmdec_corr","pmra_pmdec_corr"]

for prior in list_of_prior:
    name_chains = "Chains_5D_"+str(prior["type"])+"_loc="+str(int(prior["location"]))+"_scl="+str(int(prior["scale"]))+".h5"
    file_chains = dir_chains+name_chains
    file_csv    = file_chains.replace("h5","csv")

    if not os.path.isfile(file_chains):
        p1d = Inference(posterior=Posterior_5d,
                        prior=prior["type"],
                        prior_loc=prior["location"],
                        prior_scale=prior["scale"],
                        n_walkers=n_walkers)
        p1d.load_data(file_data,list_observables,nrows=2)
        p1d.run(n_iter,file_chains=file_chains,tol_convergence=tolerance)

    #----------------- Analysis ---------------
    a1d = Analysis(n_dim=5,file_name=file_chains,id_name=list_observables[0],
        dir_plots=dir_plots,
        tol_convergence=tolerance,
        statistic=statistic,
        transformation=None)
    a1d.plot_chains()
    a1d.save_statistics(file_csv)
#=======================================================================================