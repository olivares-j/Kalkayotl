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
import h5py
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
{"type":"Gaussian",     "location":300,"scale":80.0},
{"type":"Cauchy",       "location":300,"scale":80.0}
]

burnin_tau = 3
n_samples  = 1000
range_dist = 100,1000

#============ Directories and data =================
case       = "5D"
idx_dist   = 2
dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_ana    = dir_main  + "Analysis/NGC6774/"
dir_chains = dir_ana   + "Chains/"
file_plot  = dir_ana   + "Plots/Posteriors_"+case+".pdf"
#=======================================================================================================================

#================================== Plot posteriors ==========================================================================
pdf = PdfPages(filename=file_plot)
plt.figure(figsize=(6,6))
for j,prior in enumerate(list_of_priors):
	file_h5 = dir_chains + "Chains_"+case+"_"+str(prior["type"])+"_loc="+str(int(prior["location"]))+"_scl="+str(int(prior["scale"]))+".h5"

	f = h5py.File(file_h5, 'r')
	names = [key for key in f.keys()]
	f.close()

	n_names = len(names)
	samples = np.empty((n_names,n_samples))

	for i,name in enumerate(names):
		reader = emcee.backends.HDFBackend(file_h5,name=str(name))
		tau    = reader.get_autocorr_time(tol=0)
		burnin = int(burnin_tau*np.max(tau))
		sample = reader.get_chain(discard=burnin,flat=True)[:,idx_dist]

		samples[i] = np.random.choice(sample,n_samples)

	#---------- Plot ----------------------
	plt.hist(samples.flatten(),range=range_dist,bins=200,histtype='step',density=True,label=prior["type"])

plt.xlabel("Distance [pc]")
plt.ylabel("Density")
plt.yscale("log")
plt.legend(
	title="Prior",
	shadow = False,
	bbox_to_anchor=(0.,1.05, 1., .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 3,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()