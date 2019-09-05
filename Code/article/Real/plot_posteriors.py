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
import numpy as np
import pandas as pn
import scipy.stats as st


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def my_mode(sample):
	mins,maxs = np.min(sample),np.max(sample)
	x         = np.linspace(mins,maxs,num=1000)
	gkde      = st.gaussian_kde(sample.flatten())
	ctr       = x[np.argmax(gkde(x))]
	return ctr

#----------- Galactic prior --------
list_of_priors = [
# {"type":"EDSD",     "color":"black",  },
{"type":"Uniform",  "color":"blue",   },
{"type":"Gaussian", "color":"orange", },
{"type":"Cauchy",   "color":"green",  },
{"type":"GMM",      "color":"maroon", }
]

case = "Rup147"

burnin     = 3000
n_samples  = 1000
n_bins     = 200
range_dist = 250,350

#============ Directories and data =================

dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_out    = dir_main  + "Outputs/"
dir_chains = dir_out   + case + "/"
file_plot  = dir_out   + "Plots/Posteriors_"+case+".pdf"
#=======================================================================================================================

#================================== Plot posteriors ==========================================================================
pdf = PdfPages(filename=file_plot)
plt.figure(figsize=(6,6))

for j,prior in enumerate(list_of_priors):
	print(prior["type"])
	#-------- Read chain --------------------------------------
	file_csv = (dir_chains + prior["type"] +"/1D/chain-0.csv")
	df       = pn.read_csv(file_csv,usecols=lambda x: (("source" in x) and ("interval" not in x) and ("log" not in x)))[burnin:]
	samples = np.zeros((len(df.columns),len(df)))
	for i,name in enumerate(df.columns):
		samples[i] = df[name]

	# print(my_mode(samples.flatten()))

	#---------- Plot ----------------------
	plt.hist(samples.flatten(),bins=n_bins,range=range_dist,
		histtype='step',density=True,
		color=prior["color"],label=prior["type"])



plt.xlabel("Distance [pc]")
plt.ylabel("Density")
plt.yscale("log")
plt.ylim(bottom=1e-4)

plt.legend(
	title="Prior",
	shadow = False,
	bbox_to_anchor=(0., 1.01, 1., .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = len(list_of_priors),
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')

pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()


