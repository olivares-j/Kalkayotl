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
import numpy as np
import pandas as pn

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines

#============ Directories and data =================
dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_out    = dir_main  + "Outputs/Synthetic/Gaussian_12345/"
dir_data   = dir_main  + "Data/Synthetic/Gaussian_12345/"
file_plot  = dir_main  + "Outputs/Plots/Accuracy_and_precision.pdf"
#==================================================================

prior = "Gaussian"
#----------- prior --------
colors    = ["orange","green"]

clusters = [
			{"name":"Gaussian",     "distance":100  },
			{"name":"Gaussian",     "distance":200  },
			{"name":"Gaussian",     "distance":300  },
			{"name":"Gaussian",     "distance":400  },
			{"name":"Gaussian",     "distance":500  },
			{"name":"Gaussian",     "distance":600  },
			{"name":"Gaussian",     "distance":700  },
			{"name":"Gaussian",     "distance":800  },
			{"name":"Gaussian",     "distance":900  },
			{"name":"Gaussian",     "distance":1000 },
			{"name":"Gaussian",     "distance":1600 },
			{"name":"Gaussian",     "distance":1800 },
			{"name":"Gaussian",     "distance":2300 },
			{"name":"Gaussian",     "distance":2500 },
			{"name":"Gaussian",     "distance":4000 },
			{"name":"Gaussian",     "distance":5800 },
			]

iocs      = [
			{"name":"off","value":"indep","marker":"d","color":"brown","linestyle":"--","sft":-0.05},
			{"name":"on", "value":"corr", "marker":"o","color":"green","linestyle":"-","sft":+0.0}
			]
parameters =[
			{"name":"loc","label":"Location bias [pc]","inset_y_pos":0.5,"ylim":[-100,300]},
			{"name":"scl","label":"Scale bias [pc]",   "inset_y_pos":0.1,"ylim":[-50 ,200]},

]

statistics = [
			{"name":"Credibility [%]",   "color":"green","scale":"linear"},
			{"name":"RMS [pc]",          "color":"brown","scale":"log"},
			{"name":"Source CI [pc]",    "color":"blue", "scale":"log"},
			{"name":"Parameter CI [pc]", "color":"cyan", "scale":"log"},
			{"name":"Correlation",       "color":"red",  "scale":"linear"},
			]
#================================== Plot points ==========================================================================
n_cluster = len(clusters)
distances = np.array([cluster["distance"] for cluster in clusters],dtype="int")
sts       = np.zeros((n_cluster,5,3))
bias      = np.zeros((2,2,3,n_cluster))


for j,cluster in enumerate(clusters):
	print(40*"=")
	print(cluster["name"],cluster["distance"])
	for i,ioc in enumerate(iocs):
		#------------- Files ----------------------------------------------------------------------------------
		dir_chains    = dir_out    + cluster["name"] +"_"+str(cluster["distance"])+"/"+prior+"/"+ioc["value"]+"/" 
		file_par      = dir_chains + "Cluster_mode.csv"
		file_src      = dir_chains + "Sources_mode.csv"
		file_true     = dir_data   + cluster["name"]+"_"+str(cluster["distance"])+".csv"
		#-----------------------------------------------------------------------------------------------------

		#-------- Read data --------------------------------
		true = pn.read_csv(file_true,usecols=["ID","r","parallax_error","parallax"])
		true.sort_values(by="ID",inplace=True)
		true.set_index("ID",inplace=True)
		true_val = np.array([np.mean(true["r"]),np.std(true["r"])])
		#--------------------------------------------------

		# ------ Cluster parameters --------------------------------------------------------
		df_pars  = pn.read_csv(file_par,usecols=["mode","lower","upper"])
		df_pars.insert(loc=0,column="true",value=true_val)

		df_pars["Bias"]  = df_pars.apply(lambda x: x["mode"]  - x["true"],  axis = 1)
		df_pars["Span"]  = df_pars.apply(lambda x: x["upper"] - x["lower"], axis = 1)
		# ---------------------------------------------------------------------------------

		#------ Populate arrays -------
		bias[i,:,0,j] = df_pars["Bias"]
		bias[i,:,1,j] = df_pars["mode"] - df_pars["lower"]
		bias[i,:,2,j] = df_pars["upper"]- df_pars["mode"]

		#------- Observed sources -------------------------------------------
		infered  = pn.read_csv(file_src,usecols=["ID","mode","lower","upper"])
		infered.sort_values(by="ID",inplace=True)
		infered.set_index("ID",inplace=True)
		df       = true.join(infered,on="ID",lsuffix="_true",rsuffix="_obs")
		#------------------------------------------------------------------------

		#----------- Compute offset and uncertainty -----------------------------
		df["Bias"]   = df.apply(lambda x: x["mode"]-x["r"], axis = 1)
		df["Offset"] = df.apply(lambda x: x["r"]-true_val[0], axis = 1)
		df["Frac"]   = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
		df["Span"]   = df.apply(lambda x: x["upper"] - x["lower"], axis = 1)
		df["In"]     = df.apply(lambda x: ((x["r"]>=x["lower"]) and (x["r"]<=x["upper"])), axis = 1)


		#------------ Statistics ---------------------------------------------------------
		sts[j,0,i] = 100*np.sum(df["In"])/len(df)
		sts[j,1,i] = np.sqrt(np.mean(df["Bias"]**2))
		sts[j,2,i] = np.mean(df["Span"])
		sts[j,3,i] = np.mean(df_pars["Span"])
		sts[j,4,i] = np.corrcoef(df["Offset"],df["Bias"])[0,1]

	delta = 100.*(sts[j,:,0] - sts[j,:,1])/sts[j,:,1]
	print("Correlations|-- OFF -- |-- ON --    | -- Delta --|")
	print("Credible:   | {0:2.1f}% |   {1:2.1f}%|   {2:0.3}%  |".format(sts[j,0,0],sts[j,0,1],delta[0]))
	print("RMS:        | {0:0.3f} |   {1:0.3f} |   {2:0.3}  |".format(sts[j,1,0],sts[j,1,1],delta[1]))
	print("Span:       | {0:0.3f} |   {1:0.3f} |   {2:0.3}  |".format(sts[j,2,0],sts[j,2,1],delta[2]))
	print("Params span | {0:0.3f} |   {1:0.3f} |   {2:0.3}  |".format(sts[j,3,0],sts[j,3,1],delta[3]))
	print("Rho         | {0:0.3f} |   {1:0.3f} |   {2:0.3}  |".format(sts[j,4,0],sts[j,4,1],delta[4]))
	print("---------------------------------------")

#============================== Plots ========================================================
pdf = PdfPages(filename=file_plot)
fig, axes = plt.subplots(num=0,nrows=2, ncols=1, sharex=True,figsize=(6,12))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.05)
#--------- Plot parameter bias ----------------------------
for j,par in enumerate(parameters):
	axes[j].set_xscale("log")
	axes[j].set_yscale("symlog")
	axes[j].set_ylabel(par["label"])
	axes[j].axhline(y=0,linestyle="--",color="grey",lw=1,zorder=-1)

	for i,ioc in enumerate(iocs):
		axes[j].errorbar(distances*(1+ioc["sft"]),bias[i,j,0],
			yerr=bias[i,0,1:],
			fmt='none',ls='none',marker="o",ms=5,
			ecolor="grey",elinewidth=0.01,zorder=1,label=None)

		axes[j].scatter(distances*(1+ioc["sft"]),bias[i,j,0],
			s=15,
			c=ioc["color"],
			marker=ioc["marker"],zorder=2,label=ioc["name"])
	
axes[0].legend(
	title="Spatial correlations",
	shadow = False,
	bbox_to_anchor=(0.12,0.82, 0.75, 0.1),
	bbox_transform = fig.transFigure,
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 2,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper right'
	)
axes[1].set_xlabel("Distance [pc]")
pdf.savefig(bbox_inches='tight')
plt.close(0)


lines = [mlines.Line2D([], [],  color=ioc["color"],
								linestyle=ioc["linestyle"],
								label=ioc["name"]) for ioc in iocs]

fig, axes = plt.subplots(num=1,nrows=len(statistics), ncols=1, sharex=True,figsize=(6,12))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.0)
for i,stats in enumerate(statistics):
	for j,ioc in enumerate(iocs):
		axes[i].plot(distances,sts[:,i,j],color=ioc["color"],linestyle=ioc["linestyle"])
	axes[i].set_ylabel(stats["name"])
	axes[i].set_xscale('log')
	axes[i].set_yscale(stats["scale"])
plt.xlabel('Distance [pc]')
fig.legend(title="Spatial correlations",
	handles=lines,
	shadow = False,
	bbox_to_anchor=(0.12,0.82, 0.75, 0.1),
	bbox_transform = fig.transFigure,
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 2,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper right')
pdf.savefig(bbox_inches='tight')
plt.close(1)

pdf.close()
