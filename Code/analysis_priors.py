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

galactic_priors = [
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
{"type":"EDSD",         "location":0.0,"scale":1500.0,"color":"black",  "line_style":"--"}
]
#--------- Cluster oriented prior --------------------------------------------------------
cluster_priors = [
{"type":"Gaussian",     "scale":20.0,"color":"orange", "line_style":":"},
{"type":"Gaussian",     "scale":40.0,"color":"orange", "line_style":"-."},
{"type":"Gaussian",     "scale":80.0,"color":"orange", "line_style":"--"},
{"type":"Cauchy",       "scale":20.0,"color":"green",  "line_style":":"},
{"type":"Cauchy",       "scale":40.0,"color":"green",  "line_style":"-."},
{"type":"Cauchy",       "scale":80.0,"color":"green",  "line_style":"--"}
]

list_of_cases    = ["Cluster_50_20_random","Cluster_100_20_random","Cluster_300_20_random","Cluster_500_20_random"]
locations        = ["50","100","300","500"]
statistic        = "map"
list_observables = ["ID","parallax","parallax_error"]
id_gal_plt       = [0,3,6,9]  # From the list of galactic prior
id_cls_plt       = [0,3]      # From the list of cluster prior
id_scl_plt       = [0,1,2]    # SInce there are three scales is the same for both types of prior
ncols            = 2          # In the plots
nrows            = 2          # In the plots 


#============ Directories and data =================

dir_main   = os.getcwd()[:-4]
dir_data   = dir_main  + "Data/"
dir_ana    = dir_main  + "Analysis/Cluster/"

file_cls_plot  = dir_ana + "Cluster_cluster_priors_"+statistic+".pdf"
file_gal_plot  = dir_ana + "Cluster_galactic_priors_"+statistic+".pdf"
file_tex       = dir_ana + "Cluster_mad_"+statistic+".tex"
#=======================================================================================================================

#================================== Galactic priors ==========================================================================
pdf = PdfPages(filename=file_gal_plot)
plt.figure(0)
fig, axs = plt.subplots(nrows,ncols,figsize=(10,10),squeeze=True)
for i,case in enumerate(list_of_cases):
	idx = np.unravel_index(i,(nrows,ncols))
	dir_chains = dir_ana   + case+"/Chains/"
	file_data  = dir_data  + case + ".csv"
	data       = pn.read_csv(file_data)
	for j,prior in enumerate(galactic_priors):
		file_csv = dir_chains + "Chains_1D_"+str(prior["type"])+"_loc="+str(int(prior["location"]))+"_scl="+str(int(prior["scale"]))+"_"+statistic+".csv"

		infered  = pn.read_csv(file_csv,usecols=["ID","dist_ctr"]) 
		df   = data.join(infered,on="ID",lsuffix="_data",rsuffix="_estimate")
		idnv = np.where(df['dist']<=0)[0]
		df   = df.drop(idnv)

		df["Diff"] = df.apply(lambda x: (x["dist_ctr"] - x["dist"])/x["dist"], axis = 1)
		df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
		df = df.sort_values(by="Frac")

		mean = df.rolling(50).mean()
		
		#---------- Plot ----------------------
		# axs[idx[0],idx[1]].scatter(df["Frac"],df["Diff"],color=prior["color"],label=None)
		# axs[idx[0],idx[1]].plot(df["Frac"],df["Diff"],linestyle=prior["line_style"],color=prior["color"],label=None)
		axs[idx[0],idx[1]].plot(mean["Frac"],mean["Diff"],linestyle=prior["line_style"],lw=1,color=prior["color"],label=None)

	axs[idx[0],idx[1]].set_ylabel("Fractional error")
	axs[idx[0],idx[1]].set_xlabel("Fractional uncertainty")
	# axs[idx[0],idx[1]].yaxis.get_major_formatter().set_powerlimits((0, 1))
	axs[idx[0],idx[1]].annotate(case[:-9],xy=(0.05,0.9),xycoords="axes fraction")

prior_lines = [mlines.Line2D([], [],color=prior["color"],linestyle="-",label=prior["type"]) for prior in [galactic_priors[idx] for idx in id_gal_plt]]
scl_lines   = [mlines.Line2D([], [],color="black",linestyle=prior["line_style"],label=str(prior["scale"])+" pc") for prior in [galactic_priors[idx] for idx in id_scl_plt]]

axs[0,0].legend(handles=prior_lines,
	title="Galactic priors",
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
	bbox_to_anchor=(0., 2.25, 1., .1),
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

#=================================== Cluster prior ============================================================================
pdf = PdfPages(filename=file_cls_plot)
plt.figure(0)
fig, axs = plt.subplots(nrows,ncols,figsize=(10,10),squeeze=True)
for i,case in enumerate(list_of_cases):
	idx = np.unravel_index(i,(nrows,ncols))
	dir_chains = dir_ana   + case+"/Chains/"
	file_data  = dir_data  + case + ".csv"
	data       = pn.read_csv(file_data) 
	for j,prior in enumerate(cluster_priors):
		file_csv = dir_chains + "Chains_1D_"+str(prior["type"])+"_loc="+locations[i]+"_scl="+str(int(prior["scale"]))+"_"+statistic+".csv"
		infered  = pn.read_csv(file_csv,usecols=["ID","dist_ctr"]) 
		df   = data.join(infered,on="ID",lsuffix="_data",rsuffix="_estimate")
		idnv = np.where(df['dist']<=0)[0]
		df   = df.drop(idnv)
		
		df["Diff"] = df.apply(lambda x: (x["dist_ctr"] - x["dist"])/x["dist"], axis = 1)
		df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
		df = df.sort_values(by="Frac")
		mean = df.rolling(50).mean()
		
		#---------- Plot ----------------------
		# axs[idx[0],idx[1]].scatter(df["Frac"],df["Diff"],color=prior["color"],label=None)
		# axs[idx[0],idx[1]].plot(df["Frac"],df["Diff"],linestyle=prior["line_style"],color=prior["color"],label=None)
		axs[idx[0],idx[1]].plot(mean["Frac"],mean["Diff"],linestyle=prior["line_style"],lw=1,color=prior["color"],label=None)

	axs[idx[0],idx[1]].set_ylabel("Fractional error")
	axs[idx[0],idx[1]].set_xlabel("Fractional uncertainty")
	axs[idx[0],idx[1]].yaxis.get_major_formatter().set_powerlimits((0, 1))
	axs[idx[0],idx[1]].annotate(case[:-10],xy=(0.7,0.95),xycoords="axes fraction")

prior_lines = [mlines.Line2D([], [],color=prior["color"],linestyle="-",label=prior["type"]) for prior in [cluster_priors[idx] for idx in id_cls_plt]]
scl_lines   = [mlines.Line2D([], [],color="black",linestyle=prior["line_style"],label=str(prior["scale"])+" pc") for prior in [cluster_priors[idx] for idx in id_scl_plt]]

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


#================================ MAD =================================================
list_of_prior = sum([galactic_priors,cluster_priors],[])
MAD = np.zeros((len(list_of_prior),len(list_of_cases),2))
for i,case in enumerate(list_of_cases):
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
		idnv = np.where(df['dist']<=0)[0]
		df   = df.drop(idnv)

		df["Diff"] = df.apply(lambda x: (x["dist_ctr"] - x["dist"])/x["dist"], axis = 1)
		df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)

		df = df.sort_values(by="Frac")

		MAD[j,i,0] = np.mean(np.abs(df["Diff"]))
		MAD[j,i,1] = np.std(np.abs(df["Diff"]))

#--------- Reshape to have a 2d array
MAD = MAD.reshape((len(list_of_prior),2*len(list_of_cases)))

#------------- Names of data frame ----------------------
names_priors = [prior["type"]+"_"+str(int(prior["scale"])) for prior in list_of_prior]
names_cases  = [case[:-10] for case in list_of_cases]
col_names = pn.MultiIndex.from_product([names_cases, ["mean", "sd"]])

pn.set_option('display.float_format', '{:.2g}'.format)
df = pn.DataFrame(data=MAD,columns=col_names,index=names_priors)
df.to_latex(file_tex)

			




        

