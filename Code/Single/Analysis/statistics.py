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


#----------- prior parameters --------
list_of_priors = [
# {"type":"Uniform",      "location":0.0,"scale":1000.0},
# {"type":"Half-Gaussian","location":0.0,"scale":1000.0},
# {"type":"Half-Cauchy",  "location":0.0,"scale":1000.0},
{"type":"EDSD",	        "location":0.0,"scale":1350.0}
# {"type":"Gaussian",     "location":500.0,"scale":50.0},
# {"type":"Cauchy",       "location":500.0,"scale":50.0}
]

list_of_statistics =[
{"type":"map",   "color":"black","line_style":"-",  "name":"MAP"},
{"type":"mode",  "color":"blue","line_style":"-",   "name":"Mode"},
{"type":"median","color":"green","line_style":"-",  "name":"Median"},
{"type":"mean",  "color":"orange","line_style":"-", "name":"Mean"},
]
ylims  = [-0.01,0.1]
#============ Directories and data =================
case       = "Cluster_500"
dir_main   = os.getcwd()[:-20]
dir_data   = dir_main  + "Data/"
dir_ana    = dir_main  + "Analysis/"
dir_chains = dir_ana   + case+"/Chains/"
dir_plots  = dir_ana   + case+"/Plots/"
file_data  = dir_data  + case + "_20_random.csv"
file_plot  = dir_plots + case +"_statistics.pdf"

#------- Create directories -------
if not os.path.isdir(dir_ana):
	os.mkdir(dir_ana)
if not os.path.isdir(dir_chains):
    os.mkdir(dir_chains)
if not os.path.isdir(dir_plots):
    os.mkdir(dir_plots)
#================================================

#=======================================================================================================================
data  = pn.read_csv(file_data) 
pdf = PdfPages(filename=file_plot)
plt.figure(0)
#================================== Comparison ==========================================================================
list_observables = ["ID","parallax","parallax_error"]

for prior in list_of_priors: 
	for statistic in list_of_statistics:
		file_csv = dir_chains + "Chains_1D_"+str(prior["type"])+"_loc="+str(int(prior["location"]))+"_scl="+str(int(prior["scale"]))+"_"+statistic["type"]+".csv"

		infered  = pn.read_csv(file_csv,usecols=["ID","dist_ctr"]) 
		df       = data.join(infered,on="ID",lsuffix="_data",rsuffix="_estimate")

		df["Diff"] = df.apply(lambda x: (x["dist_ctr"] - x["dist"])/x["dist"], axis = 1)
		df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)

		df = df.sort_values(by="Frac")

		mean = df.rolling(50).mean()

		#---------- Plot ----------------------
		plt.plot(mean["Frac"],mean["Diff"],linestyle=statistic["line_style"],color=statistic["color"],label=statistic["name"])

plt.ylabel("Fractional error")
plt.xlabel("Fractional uncertainty")
plt.xscale("log")
plt.xlim(0.1,3)
plt.legend(
	shadow = False,
	bbox_to_anchor=(0.0, 1.05, 1., .0),
    borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 4,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper center')
# plt.subplots_adjust( hspace=0)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close(0)
pdf.close()

# fig, axs = plt.subplots(3, 2,figsize=(10,10), sharex=True,squeeze=True)
# for i,prior in enumerate(list_of_priors):
# 	idx = np.unravel_index(i,(3,2))
# 	for statistic in list_of_statistics:
# 		file_csv = dir_chains + "Chains_1D_"+str(prior["type"])+"_loc="+str(int(prior["location"]))+"_scl="+str(int(prior["scale"]))+"_"+statistic["type"]+".csv"

# 		infered  = pn.read_csv(file_csv,usecols=["ID","dist_ctr"]) 
# 		df   = data.join(infered,on="ID",lsuffix="_data",rsuffix="_estimate")

# 		df["Diff"] = df.apply(lambda x: (x["dist_ctr"] - x["dist"])/x["dist"], axis = 1)
# 		df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)

# 		df = df.sort_values(by="Frac")

# 		MAD = np.mean(np.abs(df["Diff"]))
# 		print(MAD)

# 		#---------- Plot ----------------------
# 		axs[idx[0],idx[1]].plot(df["Frac"],df["Diff"],linestyle=statistic["line_style"],color=statistic["color"],label=statistic["type"])

# 	axs[idx[0],idx[1]].set_ylabel("Fractional error")
# 	axs[idx[0],idx[1]].annotate(prior["type"]+" prior",xy=(0.05,0.9),xycoords="axes fraction")

# axs[2,0].set_xlabel("Fractional uncertainty")
# axs[2,1].set_xlabel("Fractional uncertainty")
# axs[0,0].legend(
# 	shadow = False,
# 	bbox_to_anchor=(0.1, 1.05, 2., .05),
#     borderaxespad=0.,
# 	frameon = True,
# 	fancybox = True,
# 	ncol = 4,
# 	fontsize = 'smaller',
# 	mode = 'expand',
# 	loc = 'upper center')
# plt.subplots_adjust( hspace=0)
# pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
# plt.close(0)
# pdf.close()

			




        

