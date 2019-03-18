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

list_of_prior = [
{"type":"Uniform",      "location":0.0,"scale":1000.0,"color":"blue",   "line_style":":"}
# {"type":"Uniform",      "location":0.0,"scale":1350.0,"color":"blue",   "line_style":"-."},
# {"type":"Uniform",      "location":0.0,"scale":1500.0,"color":"blue",   "line_style":"--"},
# {"type":"Half-Gaussian","location":0.0,"scale":1000.0,"color":"orange", "line_style":":"},
# {"type":"Half-Gaussian","location":0.0,"scale":1350.0,"color":"orange", "line_style":"-."},
# {"type":"Half-Gaussian","location":0.0,"scale":1500.0,"color":"orange", "line_style":"--"},
# {"type":"Half-Cauchy",  "location":0.0,"scale":1000.0,"color":"green",  "line_style":":"},
# {"type":"Half-Cauchy",  "location":0.0,"scale":1350.0,"color":"green",  "line_style":"-."},
# {"type":"Half-Cauchy",  "location":0.0,"scale":1500.0,"color":"green",  "line_style":"--"},
# {"type":"EDSD",	        "location":0.0,"scale":1000.0,"color":"black",  "line_style":":"},
# {"type":"EDSD",         "location":0.0,"scale":1350.0,"color":"black",  "line_style":"-."},
# {"type":"EDSD",         "location":0.0,"scale":1500.0,"color":"black",  "line_style":"--"},
]

priors = ["Uniform","Half-Gaussian","Half-Cauchy","EDSD"]
scales = [1000.,1350.,1500.]
color_priors = ["blue","orange","green","black"]
lsty_scales  = [":","-.","--"]
#============ Directories and data =================
case      = "Star_300_0"
dir_main   = os.getcwd()[:-4]
dir_data   = dir_main  + "Data/"
dir_ana    = dir_main  + "Analysis/"
dir_chains = dir_ana   + "Chains/"+case+"/"
dir_plots  = dir_ana   + "Plots/"
file_data  = dir_data  + case + ".csv"
file_plot  = dir_plots + case+"_error_vs_uncertainty_map.pdf"

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
#================================== Compariosn ==========================================================================
list_observables = ["ID","parallax","parallax_error"]
for prior in list_of_prior:
	file_csv = dir_chains + "Chains_1D_"+str(prior["type"])+"_loc="+str(int(prior["location"]))+"_scl="+str(int(prior["scale"]))+".csv"

	MAP  = pn.read_csv(file_csv,usecols=["ID","dist_ctr"]) 
	df   = data.join(MAP,on="ID",lsuffix="_data",rsuffix="_MAP")

	df["Diff"] = df.apply(lambda x: (x["dist_ctr"] - x["dist"])/x["dist"], axis = 1)
	df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)

	df = df.sort_values(by="Frac")

	MAD = np.mean(np.abs(df["Diff"]))
	print(MAD)

	# mean = df.rolling(50).mean()
	
	#---------- Plot ----------------------
	plt.plot(df["Frac"],df["Diff"],linestyle=prior["line_style"],color=prior["color"],label=None)
	# plt.plot(mean["Frac"],mean["Diff"],lw=1,color=prior["color"],,label=None)


prior_lines = [mlines.Line2D([], [],color=color_priors[i],linestyle="-",label=prior) for i,prior in enumerate(priors)]
scl_lines   = [mlines.Line2D([], [],color="black",linestyle=lsty_scales[i],label=str(int(scl))) for i,scl in enumerate(scales)]

legend = plt.legend(handles=prior_lines,title="Priors",loc='upper left')
plt.legend(handles=scl_lines,title="Scales",loc='upper center')
plt.gca().add_artist(legend)
plt.title("Star at 300 pc")
plt.xlabel("Fractional uncertainty")
plt.ylabel("Fractional error")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close(0)
pdf.close()

			




        

