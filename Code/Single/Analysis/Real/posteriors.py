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
{"type":"EDSD",     "location":0.0,   "scale":1350.0, "color":"black",  "line_style":"--"},
{"type":"Uniform",  "location":300.0, "scale":50.0,   "color":"blue",   "line_style":":"},
# {"type":"Uniform",  "location":300.0, "scale":100.0,  "color":"blue",   "line_style":":"},
{"type":"Gaussian", "location":300.0, "scale":50.0,   "color":"orange", "line_style":"-"},
# {"type":"Gaussian", "location":300.0, "scale":100.0,  "color":"orange", "line_style":":"},
{"type":"Cauchy",   "location":300.0, "scale":50.0,   "color":"green",  "line_style":"-."},
# {"type":"Cauchy",   "location":300.0, "scale":100.0,  "color":"green",  "line_style":":"}
]

list_of_cases  = [
{"type":"1D","color":"black","idx":0},
{"type":"3D","color":"blue", "idx":2},
{"type":"5D","color":"green","idx":2}
]

burnin_tau = 3
n_samples  = 100
n_bins     = 100
range_dist = 200,450

#============ Directories and data =================

dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_ana    = dir_main  + "Analysis/NGC6774/"
dir_chains = dir_ana   + "Chains/"
file_plot  = dir_ana   + "Plots/Posteriors.pdf"
#=======================================================================================================================

#================================== Plot posteriors ==========================================================================
pdf = PdfPages(filename=file_plot)
plt.figure(figsize=(6,6))
for k,case in enumerate(list_of_cases):
	for j,prior in enumerate(list_of_priors):
		file_h5 = (dir_chains + "Chains_"+case["type"]+"_"+
			str(prior["type"])+"_loc="+
			str(int(prior["location"]))+"_scl="+
			str(int(prior["scale"]))+".h5")

		f = h5py.File(file_h5, 'r')
		names = [key for key in f.keys()]
		f.close()

		n_names = len(names)
		samples = np.empty((n_names,n_samples))

		for i,name in enumerate(names):
			reader = emcee.backends.HDFBackend(file_h5,name=str(name))
			tau    = reader.get_autocorr_time(tol=0)
			burnin = int(burnin_tau*np.max(tau))
			sample = reader.get_chain(discard=burnin,flat=True)[:,case["idx"]]

			samples[i] = np.random.choice(sample,n_samples)

		#---------- Plot ----------------------
		plt.hist(samples.flatten(),range=range_dist,bins=n_bins,
			histtype='step',color=case["color"],linestyle=prior["line_style"],
			density=True,label=None)

prior_lines = [mlines.Line2D([], [],color="black",
					linestyle=prior["line_style"],
					label=prior["type"]) for prior in list_of_priors]
flv_lines   = [mlines.Line2D([], [],color=case["color"],
					linestyle="-",
					label=case["type"]) for case in list_of_cases]


plt.xlabel("Distance [pc]")
plt.ylabel("Density")
plt.yscale("log")
plt.ylim(1e-4,5e-2)

legend = plt.legend(handles=flv_lines,
	title="Flavours",
	shadow = False,
	bbox_to_anchor=(0.0, 1.01, 1., .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 3,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper right')

plt.legend(handles=prior_lines,
	title="Prior",
	shadow = False,
	bbox_to_anchor=(0., 1.1, 1., .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 4,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')

plt.gca().add_artist(legend)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()
