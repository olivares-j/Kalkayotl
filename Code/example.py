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
    along with PyAspidistra.  If not, see <http://www.gnu.org/licenses/>.
'''
#------------ LOAD LIBRARIES -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
import numpy as np

#--------------- Posteriors and inferer -------------------
from inference import Inference
from posterior_1d import Posterior as Posterior_1d
from posterior_3d import Posterior as Posterior_3d

#-------------- Chain analyser -------------
from chain_analyser import Analysis

#---------------- MCMC parameters  --------------------
n_iter    = 1000         # Number of iterations for the MCMC 
n_walkers = 10           # Number of walkers

#----------- prior parameters --------
prior        = "Uniform"#str(sys.argv[1]) #"EDSD", "Gaussian", "Uniform" or "Cauchy"
prior_loc    = 0.#float(sys.argv[2]) # Location of the prior
prior_scale  = 500.#float(sys.argv[3]) # Scale of the prior

#============ Directories and data =================
dir_main  = os.getcwd()[:-4]
dir_data  = dir_main + "Data/"
dir_expl  = dir_main + "Example/"
dir_chains= dir_expl + "Chains/"
dir_plots = dir_expl + "Plots/"
file_data = dir_data + "example.csv"

#------- Create directories -------
if not os.path.isdir(dir_expl):
	os.mkdir(dir_expl)
if not os.path.isdir(dir_chains):
    os.mkdir(dir_chains)
if not os.path.isdir(dir_plots):
    os.mkdir(dir_plots)
#==================================================

#======================= 1D Version =====================================================
list_observables = ["source_id","parallax","parallax_error"]
name_chains = "Chains_1D_"+str(prior)+"_loc="+str(int(prior_loc))+"_scl="+str(int(prior_scale))+".h5"
file_chains = dir_chains+name_chains

# p1d = Inference(posterior=Posterior_1d,prior=prior,prior_loc=prior_loc,prior_scale=prior_scale)
# p1d.load_data(file_data,list_observables,nrows=2)
# p1d.run(n_iter,n_walkers,file_chains=file_chains,tol_convergence=100)

a1d = Analysis(file_name=file_chains)

sys.exit()
#=======================================================================================

#===================== 3D Version ====================================================
list_observables = ["source_id","ra","dec","parallax",
                    "ra_error","dec_error","parallax_error",
                    "ra_dec_corr","ra_parallax_corr","dec_parallax_corr"]
name_chains = "Chains_3D_"+str(prior)+"_loc="+str(int(prior_loc))+"_scl="+str(int(prior_scale))+".h5"
file_chains = dir_chains+name_chains
p3d = Inference(posterior=Posterior_3d,prior=prior,prior_loc=prior_loc,prior_scale=prior_scale)
p3d.load_data(file_data,list_observables,nrows=2)
p3d.run(n_iter,n_walkers,file_chains=file_chains)
#===================================================================================
        

