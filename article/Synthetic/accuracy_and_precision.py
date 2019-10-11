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

prior = "Gaussian"
#----------- prior --------
colors    = ["orange","green"]

clusters = [
			{"name":"Pleiades",     "distance":200, "plot":False,"marker":"v","color":"royalblue",   "zorder":1},
			{"name":"Ruprecht_147", "distance":100, "plot":True, "marker":"v","color":"maroon",      "zorder":4},
			{"name":"NGC_1647",     "distance":600, "plot":False,"marker":"s","color":"forestgreen", "zorder":1},
			{"name":"Ruprecht_147", "distance":300, "plot":True, "marker":"o","color":"maroon",      "zorder":3},
			{"name":"Ruprecht_147", "distance":500, "plot":True, "marker":"d","color":"maroon",      "zorder":2},
			{"name":"Ruprecht_147", "distance":700, "plot":True, "marker":"s","color":"maroon",      "zorder":1},
			{"name":"Pleiades",     "distance":400, "plot":False,"marker":"d","color":"royalblue",   "zorder":1},
			{"name":"Pleiades",     "distance":600, "plot":False,"marker":"d","color":"royalblue",   "zorder":1},
			{"name":"NGC_1647",     "distance":800, "plot":False,"marker":"s","color":"forestgreen", "zorder":1},
			{"name":"NGC_1647",     "distance":1000, "plot":False,"marker":"s","color":"forestgreen","zorder":1},
			{"name":"IC_1848",      "distance":2260, "plot":False,"marker":"s","color":"orange","zorder":1},
			]

iocs      = [
			{"name":"Correlation off","value":"indep","marker":"d","sft":-10, "add":40},
			{"name":"Correlation on", "value":"corr", "marker":"s","sft":10,  "add":0}
			]

#============ Directories and data =================
dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_out    = dir_main  + "Outputs/Synthetic/"
file_plot  = dir_main  + "Outputs/Plots/Accuracy_and_precision.pdf"
#=======================================================================================================================


#================================== Plot points ==========================================================================
n_cluster = len(clusters)
plot_distances = [clusters[i] for i in np.where([cluster["plot"] for cluster in clusters])[0]]

#-------- Figure ----------------------------
pdf = PdfPages(filename=file_plot)
fig, axes = plt.subplots(num=0,nrows=2, ncols=1, sharex=True,figsize=(6,8))

for cluster in clusters:
	print(40*"=")
	print(cluster["name"],cluster["distance"])

	sts = np.zeros((4,3))
	for i,ioc in enumerate(iocs):
		#------------- Files ----------------------------------------------------------------------------------
		dir_chains    = dir_out    + cluster["name"] +"_"+str(cluster["distance"])+"/"+prior+"/"+ioc["value"]+"/" 
		file_par      = dir_chains + "Cluster_mode.csv"
		file_src      = dir_chains + "Sources_mode.csv"
		file_true     = dir_main   + "Data/Synthetic/"+cluster["name"]+"/"+cluster["name"]+"_"+str(cluster["distance"])+".csv"
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
		
		sts[0,i] = 100.*np.sum(df["In"])/len(df)
		sts[1,i] = np.sqrt(np.mean(df["Bias"]**2))
		sts[2,i] = np.mean(df["Span"])
		sts[3,i] = np.mean(df_pars["Span"])


		#--------- Plot parameter bias ----------------------------
		
		axes[1].errorbar(cluster["distance"]+ioc["sft"],df_pars["Bias"][0],
			yerr=np.reshape(par_err[:,0],(2,1)),
			fmt='none',ls='none',marker="o",ms=5,
			ecolor="grey",elinewidth=0.01,zorder=1,label=None)
		
		axes[1].scatter(cluster["distance"]+ioc["sft"],df_pars["Bias"][0],
			s=15,
			c=cluster["color"],
			marker=ioc["marker"],zorder=2)

		axes[0].errorbar(cluster["distance"]+ioc["sft"],df_pars["Bias"][1],
			yerr=np.reshape(par_err[:,1],(2,1)),
			fmt='none',ls='none',marker="o",ms=5,
			ecolor="grey",elinewidth=0.01,zorder=1,label=None)
		
		axes[0].scatter(cluster["distance"]+ioc["sft"],df_pars["Bias"][1],
			s=15,
			c=cluster["color"],
			marker=ioc["marker"],zorder=2)


		#--------- Plot source bias ----------------------------
		if cluster["plot"]:
			plt.figure(1,figsize=(6,6))
			# plt.errorbar(df["Offset"],df["Bias"],yerr=source_err,
			# 	fmt='none',ls='none',marker="o",ms=5,
			# 	ecolor="grey",elinewidth=0.01,zorder=1,label=None)
			
			plt.scatter(df["Offset"],df["Bias"]+ioc["add"],s=10,
				c=df["Frac"],
				marker=cluster["marker"],
				zorder=cluster["zorder"],
				vmin=0.0,
				vmax=0.05,
				cmap="cividis")

			plt.figure(2,figsize=(6,6))
			# plt.errorbar(df["Offset"],df["Bias"],yerr=source_err,
			# 	fmt='none',ls='none',marker="o",ms=5,
			# 	ecolor="grey",elinewidth=0.01,zorder=1,label=None)
			
			plt.scatter(df["Offset"],df["Bias"]+ioc["add"],s=10,
				c=df["Span"],
				marker=cluster["marker"],
				zorder=cluster["zorder"],
				vmin=0.0,
				vmax=0.05,
				cmap="cividis")
	sts[:,2] = 100.*(sts[:,0] - sts[:,1])/sts[:,1]
	print("Correlations|-- OFF --|-- ON --| -- Delta --|")
	print("Reliable:   |    {0:2.1f}%|   {1:2.1f}%|   {2:0.3}% |".format(sts[0,0],sts[0,1],sts[0,2]))
	print("RMS:        |    {0:0.3} |   {1:0.3} |   {2:0.3} |".format(sts[1,0],sts[1,1],sts[1,2]))
	print("Span:       |    {0:0.3} |   {1:0.3} |   {2:0.3} |".format(sts[2,0],sts[2,1],sts[2,2]))
	print("Params span |    {0:0.3} |   {1:0.3} |   {2:0.3} |".format(sts[3,0],sts[3,1],sts[3,2]))
	print("---------------------------------------")

cluster_clr = [mlines.Line2D([], [],color=cluster["color"],
								linestyle=None,
								marker="o",
								linewidth=0,
								label=cluster["name"]) for cluster in clusters[:3]]

correlation_mrk = [mlines.Line2D([], [],color="black",
								linestyle=None,
								marker=ioc["marker"],
								linewidth=0,
								label=ioc["name"]) for ioc in iocs]

fig = plt.figure(0)
legend = fig.legend(handles=correlation_mrk,
	shadow = False,
	bbox_to_anchor=(0.1,0.92, 0.8, 0),
	bbox_transform = fig.transFigure,
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 2,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper right')

fig.legend(handles=cluster_clr,
	title="Cluster",
	shadow = False,
	bbox_to_anchor=(0.1, 0.98, 0.8, 0),
	bbox_transform = fig.transFigure,
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = n_cluster,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper right')


fig.gca().add_artist(legend)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)
axes[1].set_xlabel("Distance [pc]")
axes[1].set_ylabel("Location bias [pc]")
axes[0].set_ylabel("Scale bias [pc]")
axes[0].set_xscale("log")
axes[1].set_xscale("log")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close(0)


distance_mrk = [mlines.Line2D([], [],color="grey",
								linestyle=None,
								marker=cluster["marker"],
								linewidth=0,
								label=str(cluster["distance"])) for cluster in plot_distances]
plt.figure(1)
plt.plot([-30,30],[70,10],linestyle=":",color="black",lw=1,zorder=-1)
plt.plot([-30,30],[30,-30],linestyle=":",color="black",lw=1,zorder=-1)
plt.legend(handles=distance_mrk,
	title="Distances [pc]",
	shadow = False,
	bbox_to_anchor=(0.0, 1.005, 1.0, .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 4,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')

clrb = plt.colorbar()
# clrb.set_label("Posterior width [pc]")
clrb.set_label("Fractional uncertainty")
plt.xlabel("Offset from cluster centre [pc]")
plt.ylabel("Distance bias [pc]")
plt.xlim(-30,30)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close(1)
pdf.close()
