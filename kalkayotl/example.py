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

from parallax2distance import parallax2distance

#----------------MCMC parameters  --------------------

n_iter    = 2000              # Number of iterations for the MCMC 
n_walkers = 10

#----------- prior parameters --------
prior        = str(sys.argv[1]) #"EDSD", "Gaussian", "Uniform" or "Cauchy"
prior_loc    = int(sys.argv[2]) # Location of the prior
prior_scale  = int(sys.argv[3]) # Scale of the prior


################################ DATA SET ##################################################
# Reds the data set.
# Keep the order of observables, uncertainties and correlations
# IMPORTANT put the identifier first
fdata = "/home/jromero/Desktop/Rup147/Members/members_ALL.csv"
list_observables = ["ID_member","parallax","parallax_error"]

#------- creates Analysis directory -------
dir_out = os.getcwd() + "/Example/"
if not os.path.isdir(dir_out):
	os.mkdir(dir_out)

#------ creates prior directories --------
dir_graphs = dir_out+prior+"/"+str(prior_scale)+"/"
if not os.path.isdir(dir_out+prior):
	os.mkdir(dir_out+prior)
if not os.path.isdir(dir_graphs):
	os.mkdir(dir_graphs)
#-----------------------------------

p2d = parallax2distance(prior=prior,prior_loc=prior_loc,prior_scale=prior_scale)
p2d.load_data(fdata,list_observables)
p2d.infer(n_iter,n_walkers)
p2d.analyse()

        

