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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines


#----------- Galactic prior --------
list_of_priors = [
{"type":"EDSD",     "color":"black",   "marker":"o", "line_style":"-."},
{"type":"Uniform",  "color":"blue",    "marker":"v", "line_style":"--"},
{"type":"Gaussian", "color":"orange",  "marker":"+", "line_style":"-"},
{"type":"Cauchy",   "color":"green",   "marker":"s", "line_style":":"}
]

case     = "Gauss_1"

statistic        = "mode"
list_observables = ["ID","parallax","parallax_error"]
case_plot        = ["Cluster_300"]
id_plt           = [0,1,2,3,6,7,8,11,12,13]
id_pri_plt       = [0,1,6,11]  # From the list of prior
id_scl_plt       = [0,1,2,3]    # Since there are three scales is the same for both types of prior


#============ Directories and data =================

dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_data   = dir_main  + "Data/"
dir_out    = dir_main  + "Outputs/"
dir_plots  = dir_out   + "Plots/"
file_data  = dir_data  + case + "/" +case+".csv"
file_plot  = dir_plots + "Comparison_of_priors.pdf"

#------- Data ------
data  = pn.read_csv(file_data,usecols=["ID","r","parallax","parallax_error"]) 

#=======================================================================================================================

#================================== Plots ===============================================
MAD = np.zeros((len(list_of_priors),2))
pdf = PdfPages(filename=file_plot)
plt.figure(figsize=(6,6))

for j,prior in enumerate(list_of_priors):

	dir_chains = dir_out   + case + "/" + prior["type"] + "/" + "1D/"
	file_csv   = dir_chains + "Sources_"+statistic+".csv"
	infered    = pn.read_csv(file_csv,usecols=["ID",statistic])

	#-------- Cross-identify --------------------------------------------------
	df       = data.join(infered,on="ID",lsuffix="_data",rsuffix="_chain")

	#----------- Compute frational error and uncertainty -----------------------------
	df["Diff"] = df.apply(lambda x: (x[statistic] - x["r"])/x["r"], axis = 1)
	df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
	df = df.sort_values(by="Frac")


	MAD[j,0] = np.mean(np.abs(df["Diff"]))
	MAD[j,1] = np.std(np.abs(df["Diff"]))

	mean = df.rolling(20).mean()
	
	#---------- Plot ----------------------
	plt.plot(mean["Frac"],mean["Diff"],lw=2,
				linestyle=prior["line_style"],
				color=prior["color"],
				label=prior["type"])

	
plt.xlabel("Fractional uncertainty")
plt.ylabel("Fractional error")
plt.xscale("log")
plt.ylim(-0.025,0.1)
plt.xlim(0.01,0.3)


plt.legend(
	title="Priors",
	shadow = False,
	bbox_to_anchor=(0.0, 1.1, 1.0, 0.0),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = len(list_of_priors),
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')

pdf.savefig(bbox_inches='tight')
plt.close()
pdf.close()
#================================================================================================

print(MAD)





        

