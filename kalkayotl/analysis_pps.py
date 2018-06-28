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
import scipy.stats as st
import random

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec

import pandas as pn

from astroML.decorators import pickle_results
import progressbar
from matplotlib.ticker import NullFormatter

from infer_prior import pps


####################### PARAMETERS #################################################################
#----------------Mock data and MCMC parameters  --------------------
random_state = 1234              # Random state for the synthetic data

data_loc,data_scale    = 100, 10   # Location and scale of the distribution for the mock data
data_distribution      = st.norm # Change it according to your needs

N_samples = 1000             # Number of mock distances
N_iter    = 1000              # Number of iterations for the MCMC 

#----------- prior parameters --------
prior        = str(sys.argv[1]) #"EDSD", "Gaussian", "Uniform" or "Cauchy"

#####################################################################################################

############### DIRECTORIES ###################
#------- creates Analysis directory -------
dir_out = os.getcwd() + "/Analysis/"
if not os.path.isdir(dir_out):
	os.mkdir(dir_out)

#------ creates prior directories --------
dir_graphs = dir_out+prior
if not os.path.isdir(dir_graphs):
	os.mkdir(dir_graphs)


##############################################

############################### SYNTHETIC DATA #################################################################################
#---------- create synthetic data -------------------------------------------------------------------------
true_dst = data_distribution.rvs(loc=data_loc, scale=data_scale, size=N_samples,random_state=random_state)

#---- obtains the parallax -------
pax      = map(lambda x: 1/x, true_dst)
#----- assigns an uncertainty similar to those present in TGAS data ------- 
u_pax    = st.chi2.rvs(df=2.54,loc=0.21e-3, scale=0.069e-3, size=N_samples,random_state=random_state) #Values from fit to TGAS data
ru_pax   = u_pax/pax

######################################################################################################################################

################### SAMPLE AND ANALYSIS #############################
#-------------- Initialise class --------------------
p2g = pps(pax,u_pax,nwalkers=10,distance_prior=prior,burnin_frac=0.5)

#--------------- RUN sampler ---------
sample = p2g.run(N_iter)

#-------- analyse the sample -------
quantiles = p2g.analyse_sample(sample,file_graphs=dir_graphs+"/analysis_sample.pdf")
median    = quantiles[1]

######################################################################

############### COMPARE MODEL WITH DATA ##############################
rv = data_distribution(loc=median[0],scale=median[1])
x  = np.linspace(rv.ppf(0.01)*0.9,rv.ppf(0.99)*1.1, 1000)

pdf = PdfPages(filename=dir_graphs+"/Data_and_model.pdf")
fig, ax = plt.subplots(1, 1)
ax.plot(x, rv.pdf(x),'r-', lw=2, alpha=0.6, label='Fit')
ax.hist(true_dst, bins=100, density=True, histtype='stepfilled', alpha=0.2,label="Data")
ax.legend(loc='best', frameon=False)
ax.set_xlabel("Distance [pc]")
ax.set_ylabel("Density")

pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()

print("Done!")



        

