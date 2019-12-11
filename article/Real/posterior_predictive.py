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
import scipy.stats as st

from kalkayotl.EFF import eff
from kalkayotl.King import king

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


generic_name = "Pleiades"
priors        = [
				{"family":"GMM_2","components":2,"color":["orange","green"]},
				{"family":"GMM_3","components":3,"color":["orange","green","blue"]},
				{"family":"EFF","color":"maroon"},
				{"family":"King","color":"olive"},
				]

#------------------------------------------------------------------------------------
n_samples  = 1000
n_pars     = 100
n_bins     = 100
n_cols     = 2
n_priors   = len(priors)
n_rows     = int(np.ceil(n_priors/n_cols))

rng = 80,200
x   = np.linspace(rng[0],rng[1],1000)
#============ Directories and data =================

dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_outs   = dir_main   + "Outputs/Real/"+generic_name
file_plot  = dir_main   + "Outputs/Plots/Posteriors_"+generic_name+".pdf"
#=======================================================================================================================

def chain_sources(column):
	a = "source" in column
	b = "interval" not in column
	c = "log" not in column
	return (a and b and c)

def chain_parameters(column):
	a = "loc" in column
	b = "scl" in column
	c = "weights" in column
	d = "gamma" in column
	e = "rt" in column
	f = "log" in column
	g = "stickbreaking" in column
	return ((a or b or c or d or e) and not (f or g))


#================================== Plot posteriors ==========================================================================
pdf = PdfPages(filename=file_plot)
fig, axes = plt.subplots(n_rows, n_cols,num=0,figsize=(5*n_rows,5*n_cols))
for p,prior in enumerate(priors):
	idx = np.unravel_index(p,(n_rows,n_cols))
	file_chain   = dir_outs +"/"+prior["family"]+"/" + "chain-0.csv"

	chain    = pn.read_csv(file_chain,usecols=chain_sources)[-n_samples:]
	sources  = np.zeros((len(chain.columns),len(chain)))
	for s,name in enumerate(chain.columns):
		sources[s] = chain[name]

	chain = pn.read_csv(file_chain,usecols=chain_parameters)[-n_pars:]

	if prior["family"] == "Gaussian":
		ls = chain["1D_loc__0"]
		ss = chain["1D_scl__0"]
		for l,s,g in zip(ls,ss,gs):
			density = st.norm.pdf(x,loc=l,scale=s)
			axes[idx].plot(x,density,color=prior["color"],label=None,zorder=0)

	elif "GMM" in prior["family"]:
		for i in range(prior["components"]):
			ws = chain["1D_weights__{0}".format(i)]
			ls = chain["1D_loc__{0}".format(i)]
			ss = chain["1D_scl__{0}".format(i)]

			for w,l,s in zip(ws,ls,ss):
				density = w*st.norm.pdf(x,loc=l,scale=s)
				axes[idx].plot(x,density,alpha=0.1,color=prior["color"][i],label=None,zorder=0)
	
	elif prior["family"] == "EFF":
		ls = chain["1D_loc__0"]
		ss = chain["1D_scl__0"]
		gs = chain["1D_gamma__0"]
		for l,s,g in zip(ls,ss,gs):
			density = eff.pdf(x,loc=l,scale=s,gamma=g)
			axes[idx].plot(x,density,alpha=0.1,color=prior["color"],label=None,zorder=0)

	elif prior["family"] == "King":
		ls  = chain["1D_loc__0"]
		ss  = chain["1D_scl__0"]
		rts = chain["1D_rt"]
		for l,s,rt in zip(ls,ss,rts):
			density = king.pdf(x,loc=l,scale=s,rt=rt)
			axes[idx].plot(x,density,alpha=0.1,color=prior["color"],label=None,zorder=0)


	#---------- Plot ----------------------
	axes[idx].hist(sources.flatten(),bins=n_bins,range=rng,
		histtype='step',density=True,zorder=1,
		color="black")

	axes[idx].annotate(prior["family"],xy=(0.1,0.9),xycoords="axes fraction")

	axes[idx].set_xlabel("Distance [pc]")
	axes[idx].set_ylabel("Density")
	axes[idx].set_yscale("log")
	axes[idx].set_ylim(1e-5,0.2)

# for k in range(n_clusters,n_rows*n_cols):
# 	idx = np.unravel_index(k,(n_rows,n_cols))
# 	print(idx)
# 	axes[idx].axis("off")

# handles, labels = axes[0,0].get_legend_handles_labels()
# plt.legend(handles, labels, 
# 	title="Prior",
# 	shadow = False,
# 	loc = 'upper center', 
# 	ncol=2,
# 	frameon = True,
# 	fancybox = True,
# 	fontsize = 'smaller',
# 	mode = 'expand'
# 	)

pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close(0)
pdf.close()


