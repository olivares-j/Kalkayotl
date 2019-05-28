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
{"type":"EDSD",     "scale":1350, "color":"black",    "marker":"o", "line_style":"-."},
{"type":"Uniform",  "scale":10,   "color":"blue",     "marker":"v", "line_style":":"},
{"type":"Uniform",  "scale":20,   "color":"blue",     "marker":"v", "line_style":"-"},
{"type":"Uniform",  "scale":30,   "color":"blue",     "marker":"v", "line_style":"--"},
{"type":"Uniform",  "scale":40,   "color":"blue",     "marker":"v", "line_style":":"},
{"type":"Uniform",  "scale":50,   "color":"blue",     "marker":"v", "line_style":":"},
{"type":"Gaussian", "scale":10,   "color":"orange",    "marker":"+", "line_style":":"},
{"type":"Gaussian", "scale":20,   "color":"orange",    "marker":"+", "line_style":"-"},
{"type":"Gaussian", "scale":30,   "color":"orange",    "marker":"+", "line_style":"--"},
{"type":"Gaussian", "scale":40,   "color":"orange",    "marker":"+", "line_style":":"},
{"type":"Gaussian", "scale":50,   "color":"orange",    "marker":"+", "line_style":":"},
{"type":"Cauchy",   "scale":10,   "color":"green",   "marker":"s", "line_style":":"},
{"type":"Cauchy",   "scale":20,   "color":"green",   "marker":"s", "line_style":"-"},
{"type":"Cauchy",   "scale":30,   "color":"green",   "marker":"s", "line_style":"--"},
{"type":"Cauchy",   "scale":40,   "color":"green",   "marker":"s", "line_style":":"},
{"type":"Cauchy",   "scale":50,   "color":"green",   "marker":"s", "line_style":":"}
]

list_of_cases    = [
{"name":"Cluster_300","file":"Cluster_300_20_random","location":300,"marker":"s"},
{"name":"Cluster_500","file":"Cluster_500_20_random","location":500,"marker":"o"}
]

generic_case     = "Cluster"

statistic        = "map"
list_observables = ["ID","parallax","parallax_error"]
case_plot        = ["Cluster_300"]
id_plt           = [0,1,2,3,6,7,8,11,12,13]
id_pri_plt       = [0,1,6,11]  # From the list of prior
id_scl_plt       = [0,1,2,3]    # Since there are three scales is the same for both types of prior


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
plt.figure(figsize=(6,6))
for i,case in enumerate(list_of_cases):
	dir_chains = dir_ana   + case["name"] + "/Chains/"
	file_data  = dir_data  + case["name"] + "/"+ case["file"] + ".csv"
	data       = pn.read_csv(file_data)
	for j,prior in enumerate(list_of_priors):
		if prior["type"] is "EDSD":
			location = 0
		else:
			location = case["location"]

		file_csv = (dir_chains + "Chains_1D_"+str(prior["type"])+
					"_loc="+str(location)+
					"_scl="+str(prior["scale"])+
					"_"+statistic+".csv")

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
		if case["name"] in case_plot and j in id_plt:
			plt.plot(mean["Frac"],mean["Diff"],
					linestyle=prior["line_style"],
					lw=2,
					color=prior["color"],
					label=None)

	
	plt.xlabel("Fractional uncertainty")
	plt.xscale("log")
	plt.ylim(-0.1,0.5)
	plt.xlim(0.01,1.5)

prior_lines = [mlines.Line2D([], [],color=prior["color"],
				linestyle="-",
				label=prior["type"]) for prior in [list_of_priors[idx] for idx in id_pri_plt]]

scl_lines   = [mlines.Line2D([], [],color="black",
				linestyle=prior["line_style"],
				label=str(prior["scale"])+" pc") for prior in [list_of_priors[idx] for idx in id_scl_plt]]

plt.ylabel("Fractional error")

legend = plt.legend(handles=scl_lines,
	title="Scales",
	shadow = False,
	bbox_to_anchor=(0.025, 1.01, 0.95, .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 6,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper right')

plt.legend(handles=prior_lines,
	title="Priors",
	shadow = False,
	bbox_to_anchor=(0.025, 1.1, 0.95, .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 4,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')



plt.gca().add_artist(legend)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()

plt.figure(figsize=(6,6))
for i,case in enumerate(list_of_cases):
	for j,prior in enumerate(list_of_priors):
		y     = MAD[j,i,0]
		x     = prior["scale"] +0.2*j*np.random.random()
		y_err = MAD[j,i,1]

		plt.errorbar(x,y,yerr=y_err,xerr=None,
				fmt='none',ls='none',marker="o",ms=5,
				ecolor="grey",elinewidth=0.01,zorder=1,label=None)
			
		plt.scatter(x,y,color=prior["color"],marker=case["marker"],
			s=10,zorder=2,label=prior["type"])

plt.xlabel("Scale [pc]")
plt.ylabel("| Fractional error |")
plt.xscale("log")
plt.yscale("log")
plt.xlim(9,None)
plt.ylim(1e-2,2)
prior_lines = [mlines.Line2D([], [],color=prior["color"],
				label=prior["type"]) for prior in [list_of_priors[idx] for idx in id_pri_plt]]
plt.legend(handles=prior_lines,
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
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()
sys.exit()
#--------- Reshape to have a 2d array
# MAD = MAD.reshape((len(list_of_priors),2*len(list_of_cases)))

# #------------- Names of data frame ----------------------
# names_priors = [prior["type"]+"_"+str(int(prior["scale"])) for prior in list_of_priors]
# names_cases  = [generic_case+"_"+str(loc) for loc in locations]
# col_names = pn.MultiIndex.from_product([names_cases, ["mean", "sd"]])

# pn.set_option('display.float_format', '{:.2g}'.format)
# df = pn.DataFrame(data=MAD,columns=col_names,index=names_priors)
# df.to_latex(file_tex)

			




        

