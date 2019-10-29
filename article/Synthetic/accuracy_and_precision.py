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
dir_out    = dir_main  + "Outputs/Synthetic/"
dir_data   = dir_main  + "Data/Synthetic/Gaussian_20/"
file_plot  = dir_main  + "Outputs/Plots/Accuracy_and_precision.pdf"
#==================================================================

prior = "Gaussian"
#----------- prior --------
colors    = ["orange","green"]

clusters = [
			{"name":"Gaussian",     "distance":100, "plot":False,"marker":"v",  "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":200, "plot":False,"marker":"v",  "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":300, "plot":False,"marker":"v",  "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":400, "plot":False,"marker":"v",  "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":500, "plot":False,"marker":"v",  "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":600, "plot":False,"marker":"v",  "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":700, "plot":False,"marker":"v",  "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":800, "plot":False,"marker":"v",  "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":900, "plot":False,"marker":"v",  "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":1000, "plot":False,"marker":"v", "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":1500, "plot":False,"marker":"v", "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":2000, "plot":False,"marker":"v", "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":2500, "plot":False,"marker":"v", "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":3000, "plot":False,"marker":"v", "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":3500, "plot":False,"marker":"v", "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":4000, "plot":False,"marker":"v", "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":4500, "plot":False,"marker":"v", "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":5000, "plot":False,"marker":"v", "color":"black",   "zorder":1},
			{"name":"Gaussian",     "distance":10000, "plot":False,"marker":"v","color":"black",   "zorder":1},
			]

iocs      = [
			{"name":"off","value":"indep","marker":"d","color":"brown","sft":-0.05, "add":40},
			{"name":"on", "value":"corr", "marker":"o","color":"green","sft":+0.0,  "add":0}
			]

statistics = [
			{"name":"Credibility","color":"green"},
			{"name":"RMS",        "color":"brown"},
			{"name":"95%CI",      "color":"blue"},

			]
#================================== Plot points ==========================================================================
n_cluster = len(clusters)
distances = [cluster["distance"] for cluster in clusters]
sts       = np.zeros((n_cluster,5,3))


#-------- Figure ----------------------------
pdf = PdfPages(filename=file_plot)
fig, axes = plt.subplots(num=0,nrows=2, ncols=1, sharex=True,figsize=(6,12))

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

		df_pars["Bias"]  = df_pars.apply(lambda x: x["mode"]-x["true"], axis = 1)
		df_pars["Span"]  = df_pars.apply(lambda x: x["upper"] - x["lower"], axis = 1)
		par_err  = np.vstack((df_pars["mode"]-df_pars["lower"],df_pars["upper"]-df_pars["mode"]))
		# ---------------------------------------------------------------------------------

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
		df["In"]   = df.apply(lambda x: ((x["r"]>=x["lower"]) and (x["r"]<=x["upper"])), axis = 1)
		df.sort_values(by="Frac",inplace=True,ascending=False)
		source_err  = np.vstack((df["mode"]-df["lower"],df["upper"]-df["mode"]))
		
		sts[j,0,i] = 100.*np.sum(df["In"])/len(df)
		sts[j,1,i] = np.sqrt(np.mean(df["Bias"]**2))
		sts[j,2,i] = np.mean(df["Span"])
		sts[j,3,i] = np.mean(df_pars["Span"])
		sts[j,4,i] = np.corrcoef(df["Offset"],df["Bias"])[0,1]


		#--------- Plot parameter bias ----------------------------
		
		axes[1].errorbar(cluster["distance"]*(1+ioc["sft"]),df_pars["Bias"][0],
			yerr=np.reshape(par_err[:,0],(2,1)),
			fmt='none',ls='none',marker="o",ms=5,
			ecolor="grey",elinewidth=0.01,zorder=1,label=None)
		
		axes[1].scatter(cluster["distance"]*(1+ioc["sft"]),df_pars["Bias"][0],
			s=15,
			c=ioc["color"],
			marker=ioc["marker"],zorder=2)

		axes[0].errorbar(cluster["distance"]*(1+ioc["sft"]),df_pars["Bias"][1],
			yerr=np.reshape(par_err[:,1],(2,1)),
			fmt='none',ls='none',marker="o",ms=5,
			ecolor="grey",elinewidth=0.01,zorder=1,label=None)
		
		axes[0].scatter(cluster["distance"]*(1+ioc["sft"]),df_pars["Bias"][1],
			s=15,
			c=ioc["color"],
			marker=ioc["marker"],zorder=2)

	delta = 100.*(sts[j,:,0] - sts[j,:,1])/sts[j,:,1]
	print("Correlations|-- OFF --|-- ON --   | -- Delta --|")
	print("Credible:   |{0:2.1f}%|  {1:2.1f}%|   {2:0.3}% |".format(sts[j,0,0],sts[j,0,1],delta[0]))
	print("RMS:        | {0:0.3f} |   {1:0.3f} |   {2:0.3}  |".format(sts[j,1,0],sts[j,1,1],delta[1]))
	print("Span:       | {0:0.3f} |   {1:0.3f} |   {2:0.3}  |".format(sts[j,2,0],sts[j,2,1],delta[2]))
	print("Params span | {0:0.3f} |   {1:0.3f} |   {2:0.3}  |".format(sts[j,3,0],sts[j,3,1],delta[3]))
	print("Rho         | {0:0.3f} |   {1:0.3f} |   {2:0.3}  |".format(sts[j,4,0],sts[j,4,1],delta[4]))
	print("---------------------------------------")

correlation_mrk = [mlines.Line2D([], [],color=ioc["color"],
								linestyle=None,
								marker=ioc["marker"],
								linewidth=0,
								label=ioc["name"]) for ioc in iocs]

fig = plt.figure(0)
fig.legend(title="Spatial correlations",
	handles=correlation_mrk,
	shadow = False,
	bbox_to_anchor=(0.12,0.82, 0.8, 0.1),
	bbox_transform = fig.transFigure,
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 2,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper right')


fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)
axes[1].set_xlabel("Distance [pc]")
axes[1].set_ylabel("Location bias [pc]")
axes[0].set_ylabel("Scale bias [pc]")
axes[0].set_xscale("log")
axes[1].set_xscale("log")
axes[0].set_ylim(-20,70)
axes[1].set_ylim(-100,150)
axes[0].axhline(y=0,linestyle="--",color="grey",lw=1,zorder=-1)
axes[1].axhline(y=0,linestyle="--",color="grey",lw=1,zorder=-1)
pdf.savefig(bbox_inches='tight')
plt.close(0)


lines = [mlines.Line2D([], [],  color=sts["color"],
								linestyle=None,
								label=sts["name"]) for sts in statistics]

fig = plt.figure(1)
for i,stats in enumerate(statistics):
	plt.plot(distances,sts[:,i,0],color=stats["color"],linestyle='-')
	plt.plot(distances,sts[:,i,1],color=stats["color"],linestyle='--')

plt.xscale('log')
plt.yscale('log')
fig.legend(title="Statistics",
	handles=lines,
	shadow = False,
	bbox_to_anchor=(0.12,1.0, 0.8, 0.1),
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
