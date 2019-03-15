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
import pandas as pn

#--------------- Posteriors and inferer -------------------
from inference import Inference
from posterior_1d import Posterior as Posterior_1d

#-------------- Chain analyser -------------
from chain_analyser import Analysis

#---------------- MCMC parameters  --------------------
n_iter    = 2000         # Number of iterations for the MCMC 
n_walkers = 5           # Number of walkers
tolerance = 10

#----------- prior parameters --------
priors    = ["Uniform","EDSD"]
locations = [0.]
scales    = [1000.,1350.,1500.]

#============ Directories and data =================

dir_main  = os.getcwd()[:-4]
dir_data  = dir_main + "Data/"
dir_expl  = dir_main + "Analysis/"
case      = "Gaussian_300_20"
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

#======================= 1D Version =====================================================
data   = pn.read_csv(file_data) 
truths = np.array(data['dist'])
truths = truths.reshape((len(truths),1))

list_observables = ["ID","parallax","parallax_error"]

for i,prior in enumerate(priors):
    print("="*30,prior,"="*30)
    for j,loc in enumerate(locations):
        print("-"*30,"Location ",loc,"-"*30)
        for k,scl in enumerate(scales):
            print(" "*30,"Scale ",scl," "*30)

            name_chains = "Chains_1D_"+str(prior)+"_loc="+str(int(loc))+"_scl="+str(int(scl))+".h5"
            file_chains = dir_chains+name_chains
            file_csv    = file_chains.replace("h5","csv")

            if not os.path.isfile(file_chains):
                p1d = Inference(posterior=Posterior_1d,
                                prior=prior,
                                prior_loc=loc,
                                prior_scale=scl,
                                n_walkers=n_walkers)
                p1d.load_data(file_data,list_observables)
                p1d.run(n_iter,file_chains=file_chains,tol_convergence=tolerance)

            #----------------- Analysis ---------------
            a1d = Analysis(n_dim=1,file_name=file_chains,id_name=list_observables[0],
                dir_plots=dir_plots,
                tol_convergence=tolerance)
            # a1d.plot_chains(true_values=truths,use_map=False)
            a1d.save_statistics(file_csv,use_map=True)
#=======================================================================================
        

