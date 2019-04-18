'''
Copyright 2019 Javier Olivares Romero

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
import emcee

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines


#----------- Galactic prior --------

list_of_priors = [
{"type":"Uniform",      "location":0.0,"scale":1500.0},
{"type":"Half-Gaussian","location":0.0,"scale":1500.0},
{"type":"Half-Cauchy",  "location":0.0,"scale":1500.0},
{"type":"EDSD",         "location":0.0,"scale":1500.0},
{"type":"Gaussian",     "location":500.0,"scale":80.0},
{"type":"Cauchy",       "location":500.0,"scale":80.0}
]

case      = "Star_500_0_linear"
location  = 500.0
names     = [0,4,7,11,15,18,29,36]
ncols     = 3          # In the plots
nrows     = 2          # In the plots 
burnin_tau= 3
rg_dist   = 0,4


#============ Directories and data =================

dir_main   = os.getcwd()[:-4]
dir_data   = dir_main  + "Data/"
dir_ana    = dir_main  + "Analysis/Star/"+case+"/"
dir_chains = dir_ana   + "Chains/"
file_data  = dir_data  + case + ".csv"
file_plot  = dir_ana +  "Plots/Posterior_comparison.pdf"

data         = pn.read_csv(file_data)
data.set_index("ID", inplace=True)
data["Frac"] = data.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
data = data.loc[names,"Frac"]
#=======================================================================================================================

#================================== Plot posteriors ==========================================================================
pdf = PdfPages(filename=file_plot)
plt.figure(0)
fig, axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(10,10),squeeze=True)
for j,prior in enumerate(list_of_priors):

	idx = np.unravel_index(j,(nrows,ncols))

	file_h5 = dir_chains + "Chains_1D_"+str(prior["type"])+"_loc="+str(int(prior["location"]))+"_scl="+str(int(prior["scale"]))+".h5"

	for i,name in enumerate(names):
		reader = emcee.backends.HDFBackend(file_h5,name=str(name))
		tau    = reader.get_autocorr_time(tol=0)
		burnin = int(burnin_tau*np.max(tau))
		sample = reader.get_chain(discard=burnin,flat=True)/1000.0
	
		#---------- Plot ----------------------
		# sns.distplot(sample,kde=True,norm_hist=True,ax=axs[idx[0],idx[1]],label=name)
		axs[idx[0],idx[1]].hist(sample,bins=100,range=rg_dist,histtype='step',density=True,label=np.round(data.loc[name],decimals=2))
		axs[idx[0],idx[1]].set_yscale("log")
		axs[idx[0],idx[1]].set_ylim(1e-2,3e1)
		axs[idx[0],idx[1]].set_xlabel("Distance [kpc]")
		axs[idx[0],idx[1]].annotate(prior["type"],xy=(0.6,0.9),xycoords="axes fraction")

axs[0,0].legend(
	title="Fractional uncertainty",
	shadow = False,
	bbox_to_anchor=(0., 1.02, 3., .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = len(names),
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')
plt.subplots_adjust(wspace=0, hspace=0)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close(0)
pdf.close()