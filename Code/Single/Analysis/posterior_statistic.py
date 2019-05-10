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
{"type":"EDSD",     "scale":1350.0, "color":"black",  "line_style":"-."},
{"type":"Uniform",  "scale":50.0,   "color":"blue",   "line_style":"-"},
{"type":"Uniform",  "scale":100.0,  "color":"blue",   "line_style":":"},
{"type":"Gaussian", "scale":50.0,   "color":"orange", "line_style":"-"},
{"type":"Gaussian", "scale":100.0,  "color":"orange", "line_style":":"},
{"type":"Cauchy",   "scale":50.0,   "color":"green",  "line_style":"-"},
{"type":"Cauchy",   "scale":100.0,  "color":"green",  "line_style":":"}
]

generic_case     = "Cluster"
list_of_cases    = ["Cluster_300_20_random","Cluster_500_20_random"]
locations        = ["300","500"]
statistic        = "map"
list_observables = ["ID","parallax","parallax_error"]
id_pri_plt       = [0,1,2,3]  # From the list of galactic prior
id_scl_plt       = [0,1,4]    # Since there are three scales is the same for both types of prior
ncols            = 2          # In the plots


#============ Directories and data =================

dir_main   = os.getcwd()[:-20]
dir_data   = dir_main  + "Data/"
dir_ana    = dir_main  + "Analysis/Synthetic/"

file_plot  = dir_ana + generic_case + "_priors_"+statistic+".pdf"
file_tex   = dir_ana + generic_case  + "_mad_"  +statistic+".tex"
#=======================================================================================================================

#================================== Priors ==========================================================================
MAD = np.zeros((len(list_of_priors),len(list_of_cases),2))
pdf = PdfPages(filename=file_plot)
plt.figure(0)
fig, axs = plt.subplots(1,ncols,sharey=True,figsize=(10,5),squeeze=True)
fig.subplots_adjust(wspace=0)
for i,case in enumerate(list_of_cases):
	case_name  = generic_case +"_"+ str(locations[i])+"/"
	dir_chains = dir_ana   + case_name + "Chains/"
	file_data  = dir_data  + case_name + case + ".csv"
	data       = pn.read_csv(file_data)
	for j,prior in enumerate(list_of_priors):
		if prior["type"] == "EDSD":
			location = "0"
		else:
			location = locations[i]
		file_csv = dir_chains + "Chains_1D_"+str(prior["type"])+"_loc="+location+"_scl="+str(int(prior["scale"]))+"_"+statistic+".csv"

		infered  = pn.read_csv(file_csv,usecols=["ID","dist_ctr"]) 
		df   = data.join(infered,on="ID",lsuffix="_data",rsuffix="_estimate")
		idnv = np.where(df['dist']<=0)[0]
		df   = df.drop(idnv)

		df["Diff"] = df.apply(lambda x: (x["dist_ctr"] - x["dist"])/x["dist"], axis = 1)
		df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
		df = df.sort_values(by="Frac")

		MAD[j,i,0] = np.mean(np.abs(df["Diff"]))
		MAD[j,i,1] = np.std(np.abs(df["Diff"]))

		mean = df.rolling(50).mean()
		
		#---------- Plot ----------------------
		axs[i].plot(mean["Frac"],mean["Diff"],linestyle=prior["line_style"],lw=1,color=prior["color"],label=None)

	
	axs[i].set_xlabel("Fractional uncertainty")
	axs[i].set_xscale("log")
	axs[i].set_ylim(-0.1,0.5)
	axs[i].set_xlim(0.01,1.5)
	axs[i].annotate(generic_case+"_"+str(locations[i]),xy=(0.05,0.9),xycoords="axes fraction")

prior_lines = [mlines.Line2D([], [],color=prior["color"],linestyle="-",label=prior["type"]) for prior in [list_of_priors[idx] for idx in id_pri_plt]]
scl_lines   = [mlines.Line2D([], [],color="black",linestyle=prior["line_style"],label=str(prior["scale"])+" pc") for prior in [list_of_priors[idx] for idx in id_scl_plt]]
axs[0].set_ylabel("Fractional error")
axs[0].legend(handles=prior_lines,
	title="Priors",
	shadow = False,
	bbox_to_anchor=(0.025, 1.01, 0.95, .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 4,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')

legend = plt.legend(handles=scl_lines,
	title="Scales",
	shadow = False,
	bbox_to_anchor=(0.025, 1.01, 0.95, .1),
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

#--------- Reshape to have a 2d array
MAD = MAD.reshape((len(list_of_priors),2*len(list_of_cases)))

#------------- Names of data frame ----------------------
names_priors = [prior["type"]+"_"+str(int(prior["scale"])) for prior in list_of_priors]
names_cases  = [generic_case+"_"+str(loc) for loc in locations]
col_names = pn.MultiIndex.from_product([names_cases, ["mean", "sd"]])

pn.set_option('display.float_format', '{:.2g}'.format)
df = pn.DataFrame(data=MAD,columns=col_names,index=names_priors)
df.to_latex(file_tex)

			




        

