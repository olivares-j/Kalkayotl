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
{"type":"Uniform",      "location":0.0,"scale":1000.0,"color":"blue",   "line_style":":"},
{"type":"Uniform",      "location":0.0,"scale":1350.0,"color":"blue",   "line_style":"-."},
{"type":"Uniform",      "location":0.0,"scale":1500.0,"color":"blue",   "line_style":"--"},
{"type":"Half-Gaussian","location":0.0,"scale":1000.0,"color":"orange", "line_style":":"},
{"type":"Half-Gaussian","location":0.0,"scale":1350.0,"color":"orange", "line_style":"-."},
{"type":"Half-Gaussian","location":0.0,"scale":1500.0,"color":"orange", "line_style":"--"},
{"type":"Half-Cauchy",  "location":0.0,"scale":1000.0,"color":"green",  "line_style":":"},
{"type":"Half-Cauchy",  "location":0.0,"scale":1350.0,"color":"green",  "line_style":"-."},
{"type":"Half-Cauchy",  "location":0.0,"scale":1500.0,"color":"green",  "line_style":"--"},
{"type":"EDSD",	        "location":0.0,"scale":1000.0,"color":"black",  "line_style":":"},
{"type":"EDSD",         "location":0.0,"scale":1350.0,"color":"black",  "line_style":"-."},
{"type":"EDSD",         "location":0.0,"scale":1500.0,"color":"black",  "line_style":"--"},
#--------- Cluster oriented priors --------------------------------------------------------
{"type":"Gaussian",     "scale":20.0,"color":"orange", "line_style":":"},
{"type":"Gaussian",     "scale":40.0,"color":"orange", "line_style":"-."},
{"type":"Gaussian",     "scale":80.0,"color":"orange", "line_style":"--"},
{"type":"Cauchy",       "scale":20.0,"color":"green",  "line_style":":"},
{"type":"Cauchy",       "scale":40.0,"color":"green",  "line_style":"-."},
{"type":"Cauchy",       "scale":80.0,"color":"green",  "line_style":"--"}
]

list_of_cases = ["Star_50_0_linear","Star_100_0_linear","Star_300_0_linear","Star_500_0_linear"]
locations     = ["50","100","300","500"]
statistic     = "map"
id_prior_plt  = [0,3]      # From the list of priors
id_scl_plt    = [0,1,2]    # From the list of priors
#============ Directories and data =================

dir_main   = os.getcwd()[:-4]
dir_data   = dir_main  + "Data/"
dir_ana    = dir_main  + "Analysis/"
file_plot  = dir_ana   + "Cluster_priors_"+statistic+".pdf"
file_tex   = file_plot.replace(".pdf",".tex")
#================================================

#=======================================================================================================================
pdf = PdfPages(filename=file_plot)
ncols = 2
nrows = 2
plt.figure(0)
fig, axs = plt.subplots(nrows,ncols,figsize=(10,10),squeeze=True)
#================================== Comparison ==========================================================================
list_observables = ["ID","parallax","parallax_error"]

MAD = np.zeros((len(list_of_cases),len(list_of_prior),2))

for i,case in enumerate(list_of_cases):
	idx = np.unravel_index(i,(nrows,ncols))
	dir_chains = dir_ana   + case+"/Chains/"
	file_data  = dir_data  + case + ".csv"
	data       = pn.read_csv(file_data) 
	for j,prior in enumerate(list_of_prior):
		if prior["type"] in ["Gaussian","Cauchy"]:
			file_csv = dir_chains + "Chains_1D_"+str(prior["type"])+"_loc="+locations[i]+"_scl="+str(int(prior["scale"]))+"_"+statistic+".csv"
		else:
			file_csv = dir_chains + "Chains_1D_"+str(prior["type"])+"_loc="+str(int(prior["location"]))+"_scl="+str(int(prior["scale"]))+"_"+statistic+".csv"

		infered  = pn.read_csv(file_csv,usecols=["ID","dist_ctr"]) 
		df   = data.join(infered,on="ID",lsuffix="_data",rsuffix="_estimate")

		df["Diff"] = df.apply(lambda x: (x["dist_ctr"] - x["dist"])/x["dist"], axis = 1)
		df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)

		df = df.sort_values(by="Frac")

		MAD[i,j,0] = np.mean(np.abs(df["Diff"]))
		MAD[i,j,1] = np.std(np.abs(df["Diff"]))
		# mean = df.rolling(50).mean()
		
		#---------- Plot ----------------------
		# plt.scatter(df["Frac"],df["Diff"],color=prior["color"],label=None)
		axs[idx[0],idx[1]].plot(df["Frac"],df["Diff"],linestyle=prior["line_style"],color=prior["color"],label=None)
		# axs[idx[0],idx[1]].plot(mean["Frac"],mean["Diff"],linestyle=prior["line_style"],lw=1,color=prior["color"],label=None)

	axs[idx[0],idx[1]].set_ylabel("Fractional error")
	axs[idx[0],idx[1]].set_xlabel("Fractional uncertainty")
	axs[idx[0],idx[1]].yaxis.get_major_formatter().set_powerlimits((0, 1))
	axs[idx[0],idx[1]].annotate(case[:-9],xy=(0.05,0.9),xycoords="axes fraction")

prior_lines = [mlines.Line2D([], [],color=prior["color"],linestyle="-",label=prior["type"]) for prior in [list_of_prior[idx] for idx in id_prior_plt]]
scl_lines   = [mlines.Line2D([], [],color="black",linestyle=prior["line_style"],label=str(prior["scale"])+" pc") for prior in [list_of_prior[idx] for idx in id_scl_plt]]

axs[0,0].legend(handles=prior_lines,
	title="Cluster priors",
	shadow = False,
	bbox_to_anchor=(0., 1.1, 1., .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 2,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')

legend = plt.legend(handles=scl_lines,
	title="Scales",
	shadow = False,
	bbox_to_anchor=(0., 2.30, 1., .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 3,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper right')

plt.gca().add_artist(legend)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close(0)
pdf.close()

names_priors = [prior["type"]+"_"+str(int(prior["scale"])) for prior in list_of_prior]
names_cases  = [case[:-9] for case in list_of_cases]
data_mad = pn.DataFrame(data=MAD[:,:,0].T,columns=names_cases,index=names_priors)
data_mad.to_latex(file_tex)

			




        

