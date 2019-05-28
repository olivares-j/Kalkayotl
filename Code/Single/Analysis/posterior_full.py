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


#----------- Priors --------
list_of_priors = [
{"type":"EDSD",     "scale":1350, "color":"black",   "line_style":"-."},
{"type":"Uniform",  "scale":10,   "color":"blue",    "line_style":":"},
{"type":"Uniform",  "scale":20,   "color":"blue",    "line_style":"-"},
{"type":"Uniform",  "scale":30,   "color":"blue",    "line_style":"--"},
# {"type":"Uniform",  "scale":40,   "color":"blue",    "line_style":":"},
# {"type":"Uniform",  "scale":50,   "color":"blue",    "line_style":":"},
{"type":"Gaussian", "scale":10,   "color":"orange",  "line_style":":"},
{"type":"Gaussian", "scale":20,   "color":"orange",  "line_style":"-"},
{"type":"Gaussian", "scale":30,   "color":"orange",  "line_style":"--"},
# {"type":"Gaussian", "scale":40,   "color":"orange",  "line_style":":"},
# {"type":"Gaussian", "scale":50,   "color":"orange",  "line_style":":"},
{"type":"Cauchy",   "scale":10,   "color":"green",   "line_style":":"},
{"type":"Cauchy",   "scale":20,   "color":"green",   "line_style":"-"},
{"type":"Cauchy",   "scale":30,   "color":"green",   "line_style":"--"}
# {"type":"Cauchy",   "scale":40,   "color":"green",   "line_style":":"},
# {"type":"Cauchy",   "scale":50,   "color":"green",   "line_style":":"}
]

generic_case     = "Cluster"
list_of_cases    = ["Cluster_300_20_random"]
locations        = [300]
list_observables = ["ID","parallax","parallax_error"]
id_pri_plt       = [0,1,4,7]  # From the list of galactic prior
id_scl_plt       = [0,1,2,3]    # SInce there are three scales is the same for both types of prior
burnin_tau       = 3
n_samples        = 10000
idx_dist         = 0
range_dist       = -200,600


#============ Directories and data =================

dir_main   = os.getcwd()[:-20]
dir_data   = dir_main  + "Data/"
dir_ana    = dir_main  + "Analysis/Synthetic/"

file_plot  = dir_ana + generic_case + "_priors_full.pdf"
#=======================================================================================================================

#================================== Plots ==========================================================================
pdf = PdfPages(filename=file_plot)
plt.figure(figsize=(6,6))
for i,case in enumerate(list_of_cases):

	#----------- Case directory ----------------------------------------------
	dir_chains = dir_ana   + generic_case +"_"+ str(locations[i]) + "/Chains/"

	#---- True data ---------------------
	file_data  = dir_data  + generic_case +"_"+ str(locations[i]) + "/" + case + ".csv"
	data       = pn.read_csv(file_data)

	for j,prior in enumerate(list_of_priors):
		if prior["type"] == "EDSD":
			location = 0
		else:
			location = locations[i]

		file_h5 = dir_chains + "Chains_1D_"+str(prior["type"])+"_loc="+str(location)+"_scl="+str(int(prior["scale"]))+".h5"

		f = h5py.File(file_h5, 'r')
		names = [key for key in f.keys()]
		f.close()

		n_names = len(names)
		samples = np.empty((n_names,n_samples))

		for k,name in enumerate(names):
			reader = emcee.backends.HDFBackend(file_h5,name=str(name))
			tau    = reader.get_autocorr_time(tol=0)
			burnin = int(burnin_tau*np.max(tau))
			sample = reader.get_chain(discard=burnin,flat=True)[:,idx_dist]

			samples[k] = np.random.choice(sample,n_samples)

		#---------- Plot ----------------------
		plt.hist(samples.flatten()-locations[i],
			range=range_dist,bins=100,
			histtype='step',density=True,
			linestyle=prior["line_style"],lw=1,color=prior["color"],label=None)

	#------ True data ----------
	plt.hist(data["dist"]-locations[i],
			range=range_dist,bins=100,
			histtype='step',density=True,
			linestyle="-",lw=1,color="red",label=None)

	plt.xlabel("Offset from centre [pc]")
	plt.yscale("log")
	plt.ylim(1e-4,1e-1)
	plt.xlim(range_dist)
	plt.annotate(generic_case+"_"+str(locations[i]),xy=(0.75,0.9),xycoords="axes fraction")

prior_lines = [mlines.Line2D([], [],color=prior["color"],linestyle="-",label=prior["type"]) for prior in [list_of_priors[idx] for idx in id_pri_plt]]
scl_lines   = [mlines.Line2D([], [],color="black",linestyle=prior["line_style"],label=str(prior["scale"])+" pc") for prior in [list_of_priors[idx] for idx in id_scl_plt]]
plt.ylabel("Density")


legend = plt.legend(handles=scl_lines,
	title="Scales",
	shadow = False,
	bbox_to_anchor=(0.025, 1.005, 0.95, .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = len(id_scl_plt),
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper right')

plt.legend(handles=prior_lines,
	title="Priors",
	shadow = False,
	bbox_to_anchor=(0.025, 1.1, 0.95, .1),
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
