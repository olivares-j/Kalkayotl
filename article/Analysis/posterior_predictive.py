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


generic_name = "King_500_1"
distances = [100,200,300,500]
prior     = "King"
scale     = 10.0
rt        = 5.0
gamma     = 3.0


#------------------------------------------------------------------------------------
n_samples  = 1000
n_bins     = 100
n_dist     = len(distances)
n_cols     = 2
n_rows     = int(np.ceil(n_dist/n_cols))
print(n_rows,n_cols)
#============ Directories and data =================

dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_outs   = dir_main   + "Outputs/Synthetic/"+generic_name
dir_data   = dir_main   + "Data/Synthetic/"+generic_name
file_plot  = dir_main   + "Outputs/Plots/Posteriors_"+prior+".pdf"
#=======================================================================================================================

#================================== Plot posteriors ==========================================================================
pdf = PdfPages(filename=file_plot)
fig, axes = plt.subplots(n_rows, n_cols,num=0,figsize=(5*n_rows,5*n_cols))
for d,distance in enumerate(distances):

	idx = np.unravel_index(d,(n_rows,n_cols))
	file_samples   = dir_outs +"/"+prior+"_"+str(distance)+"/"+prior+ "/corr/" + "chain-0.csv"
	file_data      = dir_data +"/"+prior+"_"+str(distance)+".csv"
	print(file_samples)

	rng = distance-30,distance+30
	x   = np.linspace(rng[0],rng[1],1000)
	if prior == "Gaussian":
		axes[idx].plot(x,st.norm.pdf(x,loc=distance,scale=scale),color="red",label="True")
	elif prior == "King":
		axes[idx].plot(x,king.pdf(x,loc=distance,scale=scale,rt=rt),color="red",label="True")
	elif prior=="EFF":
		axes[idx].plot(x,eff.pdf(x,loc=distance,scale=scale,gamma=gamma),color="red",label="True")

	true = pn.read_csv(file_data,usecols=["r"]).to_numpy()
	axes[idx].hist(true.flatten(),bins=n_bins,range=rng,histtype='step',density=True,
													color="blue",label="Posterior")

	if os.path.exists(file_samples):
		df       = pn.read_csv(file_samples,usecols=lambda x: (("source" in x) and ("interval" not in x) and ("log" not in x)))[-n_samples:]
		samples  = np.zeros((len(df.columns),len(df)))
		for s,name in enumerate(df.columns):
			samples[s] = df[name]

		#---------- Plot ----------------------
		axes[idx].hist(samples.flatten(),bins=n_bins,range=rng,
			histtype='step',density=True,
			color="black",label="Posterior")

	axes[idx].set_xlabel("Distance [pc]")
	axes[idx].set_ylabel("Density")
	axes[idx].set_yscale("log")
	axes[idx].set_ylim(bottom=1e-3)

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


