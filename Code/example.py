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

#----------------MCMC parameters  --------------------
n_iter    = 2000         # Number of iterations for the MCMC 
n_walkers = 10           # Number of walkers

#----------- prior parameters --------
prior        = str(sys.argv[1]) #"EDSD", "Gaussian", "Uniform" or "Cauchy"
prior_loc    = int(sys.argv[2]) # Location of the prior
prior_scale  = int(sys.argv[3]) # Scale of the prior

#============ Directories and data =================
dir_current = os.getcwd()
file_data = dir_current+ "/Data/members_ALL.csv"

#------- Create directories -------
dir_main   = dir_current + "/Example/"
dir_chains = dir_main +"Chains/"
dir_plots  = dir_main +"Plots/"
if not os.path.isdir(dir_main):
	os.mkdir(dir_main)
if not os.path.isdir(dir_main):
    os.mkdir(dir_chains)
if not os.path.isdir(dir_main):
    os.mkdir(dir_plots)
#==================================================


#======================= 1D Version =====================================================
name_chains_1d   = "Chains_1D_"+str(prior)+"_"+str(prior_loc)+"_"+str(prior_scale)+".h5"
list_observables = ["ID_member","parallax","parallax_error"]

p1d = Inference(Posterior_1d(prior=prior,
                            prior_loc=prior_loc,
                            prior_scale=prior_scale))

p1d.load_data(fdata,list_observables)
p1d.infer(n_iter,n_walkers,file_chains=dir_chains+name_chains_1d)
p1d.analyse()
#=======================================================================================

#===================== 3D Version ====================================================
name_chains_3d   = "Chains_1D_"+str(prior)+"_"+str(prior_loc)+"_"+str(prior_scale)+".h5"
list_observables = ["ID_member","parallax","parallax_error"]
p3d = Inference(Pos3d(prior=prior,prior_loc=prior_loc,prior_scale=prior_scale))
p3d.load_data(fdata,list_observables)
p3d.infer(n_iter,n_walkers)
p3d.analyse()
#===================================================================================
        

