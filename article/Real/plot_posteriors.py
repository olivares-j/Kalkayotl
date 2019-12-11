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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def my_mode(sample):
	mins,maxs = np.min(sample),np.max(sample)
	x         = np.linspace(mins,maxs,num=1000)
	gkde      = st.gaussian_kde(sample.flatten())
	ctr       = x[np.argmax(gkde(x))]
	return ctr

#-----------prior ----------------------------------------------------------------

list_of_clusters = [
{"name":"Pleiades",    "distance":136, "delta":100},
{"name":"Pleiades",    "distance":136, "delta":100},
{"name":"Pleiades",    "distance":136, "delta":100},
{"name":"Pleiades",    "distance":136, "delta":100},
# {"name":"Ruprecht_147","distance":305, "delta":80},
# {"name":"NGC_1647",    "distance":590, "delta":120},
# # {"name":"NGC_2264",    "distance":730, "delta":200},
# # {"name":"NGC_2682",    "distance":860, "delta":200},
# {"name":"NGC_2244",    "distance":1800, "delta":1000},
# # {"name":"NGC_188",     "distance":1860, "delta":500},
# {"name":"IC_1848",     "distance":2260, "delta":1000},
# {"name":"NGC_2420",    "distance":2500, "delta":1000},
# {"name":"NGC_6791",    "distance":4500, "delta":2000},
# {"name":"NGC_3603",    "distance":7000, "delta":5000},
]

priors = [
# {"name":"EDSD",     "color":"black"  },
# {"name":"Uniform",  "color":"blue"   },
# {"name":"Gaussian", "color":"orange" },
# {"name":"Cauchy",   "color":"pink"   },
{"name":"GMM",      "color":"green"  },
# {"name":"EFF",      "color":"purple" },
# {"name":"King",     "color":"olive"  }
]


#------------------------------------------------------------------------------------

n_samples  = 1000
n_bins     = 200
n_clusters = len(list_of_clusters)
n_cols     = 2
n_rows     = int(np.ceil(n_clusters/n_cols))
print(n_rows,n_cols)
#============ Directories and data =================

dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_data   = dir_main   + "Outputs/Real/"
file_plot  = dir_main   + "Outputs/Plots/Posteriors.pdf"
#=======================================================================================================================

#================================== Plot posteriors ==========================================================================
pdf = PdfPages(filename=file_plot)
fig, axes = plt.subplots(n_rows, n_cols,num=0,figsize=(10*n_rows,5*n_cols))
for k,cluster in enumerate(list_of_clusters):
	dir_cluster = dir_data   + cluster["name"] + "/"

	print(10*"="+cluster["name"]+10*"=")

	idx = np.unravel_index(k,(n_rows,n_cols))
	print(idx)

	rng = cluster["distance"]-cluster["delta"],cluster["distance"]+cluster["delta"]
	x   = np.linspace(rng[0],rng[1],1000)

	#----- Loop over prior ------
	for prior in priors:
		#-------- Directory--------------------------------------
		dir_chains = dir_cluster + prior["name"] + "/"

		if os.path.exists(dir_chains):
			print(prior["name"])
			file_csv =  dir_chains + "chain-0.csv"
			df       = pn.read_csv(file_csv,usecols=lambda x: (("source" in x) and ("interval" not in x) and ("log" not in x)))[-n_samples:]
			samples  = np.zeros((len(df.columns),len(df)))
			for s,name in enumerate(df.columns):
				samples[s] = df[name]

			#---------- Plot ----------------------
			axes[idx].hist(samples.flatten(),bins=n_bins,range=rng,
				histtype='step',density=True,
				color=prior["color"],label=prior["name"])

	axes[idx].set_xlabel("Distance [pc]")
	axes[idx].annotate(cluster["name"],xy=(0.01,0.9),xycoords="axes fraction")
	axes[idx].set_ylabel("Density")
	axes[idx].set_yscale("log")
	axes[idx].set_ylim(bottom=1e-5)

for k in range(n_clusters,n_rows*n_cols):
	idx = np.unravel_index(k,(n_rows,n_cols))
	print(idx)
	axes[idx].axis("off")

handles, labels = axes[0,0].get_legend_handles_labels()
plt.legend(handles, labels, 
	title="Prior",
	shadow = False,
	loc = 'upper center', 
	ncol=2,
	frameon = True,
	fancybox = True,
	fontsize = 'smaller',
	mode = 'expand'
	)

pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close(0)
pdf.close()


