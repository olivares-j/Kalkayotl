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

dir_main  = os.getcwd().replace("Code/article/Real","")
sys.path.insert(0,dir_main)

# from Code.EFF import eff
# from Code.King import king


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def my_mode(sample):
	mins,maxs = np.min(sample),np.max(sample)
	x         = np.linspace(mins,maxs,num=1000)
	gkde      = st.gaussian_kde(sample.flatten())
	ctr       = x[np.argmax(gkde(x))]
	return ctr

#-----------prior ----------------------------------------------------------------

list_of_priors = [
# {"type":"EDSD",     "color":"black",  },
{"type":"Uniform",  "loc":590.0,"scl":28.5,"color":"blue"   },
{"type":"Gaussian", "loc":589.9,"scl":16.8, "color":"orange" },
# # {"type":"Cauchy",   "loc":304.4,"scl":2.6, "color":"pink" },
# {"type":"GMM",      "loc":305.2,"scl":7.7, "color":"green"  },
# {"type":"EFF",      "loc":587.1,"scl":16.6, "color":"purple", "extra":3.6  },
# {"type":"King",     "loc":589.2,"scl":46.9, "color":"olive" , "extra":50.4 }
]

case = "NGC_3603"
dst  = 5993.0
dlt  = 5000.

# list_of_priors = [
# # {"type":"EDSD",     "color":"black",  },
# {"type":"Uniform",  "loc":305.1,"scl":13.6,"color":"blue"   },
# {"type":"Gaussian", "loc":304.9,"scl":7.8, "color":"orange" },
# {"type":"Cauchy",   "loc":304.4,"scl":2.6, "color":"pink" },
# {"type":"GMM",      "loc":305.2,"scl":7.7, "color":"green"  },
# {"type":"EFF",      "loc":304.8,"scl":8.7, "color":"purple", "extra":3.7  },
# {"type":"King",     "loc":305.0,"scl":9.4, "color":"olive" , "extra":27.3 }
# ]

# case = "Ruprecht_147"
# dst  = 305.0
# dlt  = 50.

# list_of_priors = [
# # {"type":"EDSD",     "color":"black",  },
# {"type":"Uniform",  "loc":135.5,"scl":5.3,"color":"blue"   },
# {"type":"Gaussian", "loc":135.7,"scl":2.8, "color":"orange" },
# {"type":"Cauchy",   "loc":135.6,"scl":1.2, "color":"pink" },
# {"type":"GMM",      "loc":135.6,"scl":2.9, "color":"green"  },
# {"type":"EFF",      "loc":135.6,"scl":5.0, "color":"purple", "extra":5.6  },
# {"type":"King",     "loc":135.5,"scl":4.7, "color":"olive" , "extra":9.7 }
# ]

# case = "Pleiades"
# dst  = 135.0
# dlt  = 20.
#------------------------------------------------------------------------------------

n_samples  = 1000
n_bins     = 200
#============ Directories and data =================

dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_out    = dir_main  + "Outputs/"
dir_chains = dir_out   + case + "/"
file_plot  = dir_out   + "Plots/Posteriors_"+case+".pdf"
#=======================================================================================================================

#================================== Plot posteriors ==========================================================================
pdf = PdfPages(filename=file_plot)
plt.figure(figsize=(6,6))

x = np.linspace(dst-dlt,dst+dlt,1000)
for j,prior in enumerate(list_of_priors):
	print(prior["type"])
	#-------- Read chain --------------------------------------
	file_csv = (dir_chains + prior["type"] +"/chain-0.csv")
	df       = pn.read_csv(file_csv,usecols=lambda x: (("source" in x) and ("interval" not in x) and ("log" not in x)))[-n_samples:]
	samples = np.zeros((len(df.columns),len(df)))
	for i,name in enumerate(df.columns):
		samples[i] = df[name]

	#---------- Plot ----------------------
	plt.hist(samples.flatten(),bins=n_bins,range=(dst-dlt,dst+dlt),
		histtype='step',density=True,
		color=prior["color"],label=prior["type"])
	# if prior["type"] is "Uniform":
	# 	plt.plot(x,st.uniform.pdf(x,loc=prior["loc"]-prior["scl"],scale=2*prior["scl"]),linestyle="--",color=prior["color"])
	# if prior["type"] is "Gaussian":
	# 	plt.plot(x,st.norm.pdf(x,loc=prior["loc"],scale=prior["scl"]),linestyle="--",color=prior["color"])
	# if prior["type"] is "Cauchy":
	# 	plt.plot(x,st.cauchy.pdf(x,loc=prior["loc"],scale=prior["scl"]),linestyle="--",color=prior["color"])
	# if prior["type"] is "EFF":
	# 	plt.plot(x,eff.pdf(x,r0=prior["loc"],rc=prior["scl"],gamma=prior["extra"]),linestyle="--",color=prior["color"])
	# if prior["type"] is "King":
	# 	plt.plot(x,king.pdf(x,r0=prior["loc"],rc=prior["scl"],rt=prior["extra"]),linestyle="--",color=prior["color"])

plt.xlabel("Distance [pc]")
plt.ylabel("Density")
plt.yscale("log")
plt.ylim(bottom=1e-5)

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


