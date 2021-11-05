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
import scipy.stats as st

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines

plt.rcParams.update({'font.size': 20})
figsize = (14,14)

loc = 500
#----------- Prior --------
list_of_priors = [
{"type":"Uniform",  "color":"green",   "plot":"corr", "statistic":"mean"},
{"type":"Gaussian", "color":"blue",    "plot":"corr", "statistic":"mean"},
{"type":"GMM",      "color":"maroon",  "plot":"corr", "statistic":"mean"},
{"type":"EFF",      "color":"violet",  "plot":"corr", "statistic":"mean"},
{"type":"King",     "color":"cyan",    "plot":"corr", "statistic":"mean"},
{"type":"EDSD",     "color":"black",   "plot":"indep","statistic":"mode"},
]

list_of_cases = [
{"name":"Gaussian","location":loc,"scale":10.0,"linestyle":"-", "linewidth":1, "type":"corr" },
{"name":"Gaussian","location":loc,"scale":10.0,"linestyle":"--","linewidth":1, "type":"indep"},
]

n_samples  = 1000
n_bins     = 30
range_dist = -50,50
range_bias = -35,35
window     = 20
xs = np.linspace(range_dist[0],range_dist[1],1000)
bias = np.empty((len(list_of_cases),len(list_of_priors),3))
z    = np.empty((len(list_of_cases),len(list_of_priors),3))


list_observables = ["ID","r","parallax","parallax_error"]


#============ Directories and data =================

dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_data   = dir_main  + "Data/Synthetic/Gaussian_500_1/"
dir_out    = dir_main  + "Outputs/Synthetic/Comparison/"
dir_plots  = dir_main  + "Outputs/Plots/"
file_plot  = dir_plots + "Comparison_of_priors.pdf"
file_tex   = dir_plots + "Table_rms_bias.tex"

#================================== Plots ===============================================
pdf = PdfPages(filename=file_plot)
#--------- Plot bias ----------------------------
fig, axes = plt.subplots(num=2,nrows=3, ncols=2, sharex=True,sharey=True,figsize=figsize)
for c,case in enumerate(list_of_cases):

	file_data  = dir_data  + case["name"] + "_" + str(case["location"])+".csv"
	#------- Data ------
	data  = pn.read_csv(file_data,usecols=list_observables) 
	data.sort_values(by="ID",inplace=True)
	data.set_index("ID",inplace=True)
	mean_dist  = np.mean(data["r"])
	#=======================================================================================================================

	plt.figure(1,figsize=figsize)
	plt.hist(data["r"]-loc,linewidth=1,bins=n_bins,range=range_dist,
				histtype='step',density=True,linestyle=":",
				color="red",label="True")

	for j,prior in enumerate(list_of_priors):
		print(30*"-")
		print(prior["type"])

		#----------- Files --------------------------------------------------------------
		dir_chains = dir_out    + case["name"] + "_" + str(case["location"]) + "/" + prior["type"] + "/" + case["type"]+"/"
		# file_cls   = dir_chains + "Cluster_"+prior["statistic"]+".csv"
		file_stats = dir_chains + "Sources_"+prior["statistic"]+".csv"
		file_Z     = dir_chains + "Cluster_Z.csv"
		#-------------------------------------------------------------------------------

		#--------------------- Evidence ----------------------------------------
		if case["type"] is "corr" and prior["type"] is not "EDSD":
			evidence = pn.read_csv(file_Z)
			evidence.set_index("Parameter",inplace=True)
			z[c,j] = evidence.loc["logZ"]
			print("Evidence: {0:5.2f}+{1:2.2f}-{2:2.2f}".format(z[c,j,1],
				z[c,j,2]-z[c,j,1],z[c,j,1]-z[c,j,0]))
		#-----------------------------------------------------------------------

		#--------------------- Statistic ---------------------------------
		infered  = pn.read_csv(file_stats,usecols=["ID",prior["statistic"],"lower","upper"])
		infered.sort_values(by="ID",inplace=True)
		infered.set_index("ID",inplace=True)

		#-------- Cross-identify --------------------------------------------------
		df    = data.join(infered,on="ID",lsuffix="_data",rsuffix="_chain")
		#--------------------------------------------------------------------------

		#----------- Compute fractional error and uncertainty -----------------------------
		df["Diff"] = df.apply(lambda x: (x[prior["statistic"]] - x["r"])/x["r"], axis = 1)
		df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
		df = df.sort_values(by="Frac")
		#------------------------------------------------------------------
		print("MAD: {1:0.4f}".format(prior["type"],np.mean(np.abs(df["Diff"]))))
		mean = df.rolling(window).mean()
		#-------------------------------------------------------------------------------

		#----------- Compute offset and uncertainty -------------------------------------------------
		df["Bias"]   = df.apply(lambda x: x[prior["statistic"]]-x["r"], axis = 1)
		df["Offset"] = df.apply(lambda x: x["r"]-mean_dist, axis = 1)
		y_err = np.vstack((df[prior["statistic"]]-df["lower"],df["upper"]-df[prior["statistic"]]))

		df["In"]   = df.apply(lambda x: ((x["r"]>=x["lower"]) and (x["r"]<=x["upper"])), axis = 1)

		print("Good distances:{0:2.1f}".format(100.*np.sum(df["In"])/len(df)))
		print("RMS: {1:0.4f}".format(prior["type"],np.sqrt(np.mean(df["Bias"]**2))))
		print("Rho: {1:0.4f}".format(prior["type"],
			np.corrcoef(df["Offset"],df["Bias"])[0,1]))

		bias[c,j,0] = np.sqrt(np.mean(df.loc[df["Frac"]<0.05,"Bias"]**2))
		bias[c,j,1] = np.sqrt(np.mean(df.loc[(df["Frac"]>0.05)&(df["Frac"]<0.1),"Bias"]**2))
		bias[c,j,2] = np.sqrt(np.mean(df.loc[df["Frac"]>0.1,"Bias"]**2))
		#-----------------------------------------------------------------------------------------


		#---------- Plot fractional error ----------------------
		plt.figure(0,figsize=(10,10))
		plt.plot(mean["Frac"],mean["Diff"],lw=case["linewidth"],
					linestyle=case["linestyle"],
					color=prior["color"],
					label=prior["type"])

		#---------- Plot posteriors----------------------
		if prior["plot"] is case["type"]:
			#----------------------------------------------------------------------------------
			plt.figure(1)	
			plt.hist(infered[prior["statistic"]]-mean_dist,bins=n_bins,range=range_dist,
				histtype='step',density=True,linewidth=case["linewidth"],
				linestyle=case["linestyle"],
				color=prior["color"])

			if j <6:
				#--------- Plot bias ----------------------------
				if j <3:
					k = 0
					l = j
				else:
					k = 1
					l = j-3
				axes[l,k].plot(range_bias,np.flip(range_bias),color="grey",linestyle="--",zorder=0)
				axes[l,k].errorbar(df["Offset"],df["Bias"],yerr=y_err,
					fmt='none',ls='none',marker="o",ms=5,
					ecolor="grey",elinewidth=0.01,zorder=1,label=None)
				
				points = axes[l,k].scatter(df["Offset"],df["Bias"],s=20,c=df["Frac"],
					zorder=2,
					vmax=0.1,
					cmap="viridis")
				axes[l,k].set_ylim(range_bias[0],range_bias[1])
				axes[l,k].set_xlim(range_bias[0],range_bias[1])
				axes[l,k].annotate(prior["type"],xy=(0.05,0.1),xycoords="axes fraction")
				axes[l,k].annotate("$\\rho$={0:0.2f}".format(np.corrcoef(df["Offset"],df["Bias"])[0,1]),
												xy=(0.99,0.9),xycoords="axes fraction",ha="right")


priors_hdl = [mlines.Line2D([], [],color=prior["color"],
								linestyle='-',
								linewidth=2,
								label=prior["type"]) for prior in list_of_priors]
plt.figure(0)	
plt.xlabel("Parallax fractional uncertainty")
plt.ylabel("Distance fractional error")
plt.xscale("log")
plt.ylim(-0.02,0.25)
locs = [0.02,0.05,0.1,0.2]
labs = [str(loc) for loc in locs]
plt.xticks(locs,labs)
plt.legend(
	title="Priors",
	handles=priors_hdl,
	shadow = False,
	bbox_to_anchor=(0.0, 1.15, 1.0, 0.0),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 3,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')
pdf.savefig(bbox_inches='tight')
plt.close()

plt.figure(1)
plt.vlines(x=0,ymin=0,ymax=1,colors=['grey'],linestyles=['--'],zorder=0)
plt.xlabel("Offset from centre [pc]")
plt.ylabel("Density")
plt.yscale("log")
# plt.ylim(5e-5,5e-2)
plt.ylim(1e-3,0.1)
plt.xlim(range_dist)
plt.legend(
	title="Priors",
	handles=priors_hdl,
	shadow = False,
	bbox_to_anchor=(0.0, 1.15, 1.0, 0.0),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 3,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')
pdf.savefig(bbox_inches='tight')
plt.close()



plt.figure(2,figsize=figsize)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)
clrb = fig.colorbar(points,orientation="horizontal",pad=0.07,ax=axes)
clrb.set_label("Parallax fractional uncertainty")
for i in range(2):
	axes[2,i].set_xlabel("Offset from centre [pc]")
for i in range(3):
	axes[i,0].set_ylabel("Distance bias [pc]")
pdf.savefig(bbox_inches='tight')
plt.close()
pdf.close()
#================================================================================================

#=============== Table of bias rms for prior and uncertainty regime ==========================
print(30*"-")
print("Printing table of rms bias ... ")
with open(file_tex, "w") as tex_file:
	header = ["Prior","$f_\\varpi$<0.05","0.05<$f_\\varpi$<0.1","$f_\\varpi$>0.1","log Z"]
	print("\\begin{tabular}{ccccc}", file=tex_file)
	print("\\hline", file=tex_file)
	print("\\hline", file=tex_file)
	print("  &  ".join(header) + "  \\\\", file=tex_file)
	print("\\hline", file=tex_file)
	for j,prior in enumerate(list_of_priors):
		str_cols = [prior["type"]]
		for i in range(3):
			str_cols.append("{0:2.2f}({1:2.2f})".format(bias[0,j,i],bias[1,j,i]))
		str_cols.append("{0:2.2f}\\pm{1:2.2f}".format(z[0,j,1],z[0,j,2]-z[0,j,1],z[0,j,1]-z[0,j,0]))
		print("  &  ".join(str_cols) + "  \\\\", file=tex_file)
	print("\\hline", file=tex_file)
	print("\\end{tabular}", file=tex_file)
print("Check file: ",file_tex)




        

