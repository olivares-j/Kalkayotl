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


#----------- Prior --------
list_of_priors = [
{"type":"EDSD",     "color":"black",   "marker":"o", "linestyle":"-"},
{"type":"Uniform",  "color":"blue",    "marker":"v", "linestyle":"-"},
{"type":"Gaussian", "color":"orange",  "marker":"+", "linestyle":"-"},
{"type":"Cauchy",   "color":"green",   "marker":"s", "linestyle":"-"},
{"type":"GMM",      "color":"maroon",  "marker":"h", "linestyle":"-"},
{"type":"EFF",      "color":"violet",  "marker":"d", "linestyle":"-"},
{"type":"King",     "color":"cyan",    "marker":"^", "linestyle":"-"},
]

case     = "Ruprecht_147"
distance = 500
scl      = 8.0

burnin     = 5000
n_samples  = 1000
n_bins     = 200
range_dist = -75,75

statistic        = "mode"
list_observables = ["ID","r","parallax","parallax_error"]


#============ Directories and data =================

dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_data   = dir_main  + "Data/Synthetic/"
dir_out    = dir_main  + "Outputs/Synthetic/"
dir_plots  = dir_main  + "Outputs/Plots/"
file_data  = dir_data  + case + "/" +case+"_"+str(distance)+".csv"
file_plot  = dir_plots + "Comparison_of_priors.pdf"
file_tex   = dir_plots + "Table_rms_bias.tex"

#------- Data ------
data  = pn.read_csv(file_data,usecols=list_observables) 
data.sort_values(by="ID",inplace=True)
data.set_index("ID",inplace=True)
mean_dist  = np.mean(data["r"])

#=======================================================================================================================


bias = np.empty((len(list_of_priors),3))
#================================== Plots ===============================================
pdf = PdfPages(filename=file_plot)
for j,prior in enumerate(list_of_priors):
	print(30*"-")
	print(prior["type"])

	#----------- Files --------------------------------------------------------------
	dir_chains = dir_out    + case + "_"+str(distance)+"/" + prior["type"] + "/" + "indep/"
	file_stats = dir_chains + "Sources_"+statistic+".csv"
	file_chain = dir_chains + "chain-0.csv"
	#-------------------------------------------------------------------------------


	#--------------------- Statistic ---------------------------------
	infered  = pn.read_csv(file_stats,usecols=["ID","mode","lower","upper"])
	infered.sort_values(by="ID",inplace=True)
	infered.set_index("ID",inplace=True)

	#-------- Cross-identify --------------------------------------------------
	df    = data.join(infered,on="ID",lsuffix="_data",rsuffix="_chain")
	#--------------------------------------------------------------------------

	#----------------- Red chains -------------------------------------------------
	chain = pn.read_csv(file_chain,usecols=lambda x: (("source" in x) 
						and ("interval" not in x) and ("log" not in x)))[burnin:]
	samples = np.zeros((len(chain.columns),len(chain)))
	for i,name in enumerate(chain.columns):
		samples[i] = chain[name]
	#-----------------------------------------------------------------------------

	#----------- Compute fractional error and uncertainty -----------------------------
	df["Diff"] = df.apply(lambda x: (x[statistic] - x["r"])/x["r"], axis = 1)
	df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
	df = df.sort_values(by="Frac")
	#------------------------------------------------------------------
	print("MAD: {1:0.4f}".format(prior["type"],np.mean(np.abs(df["Diff"]))))
	mean = df.rolling(10).mean()
	#-------------------------------------------------------------------------------

	#----------- Compute offset and uncertainty -----------------------------
	df["Bias"]   = df.apply(lambda x: x["mode"]-x["r"], axis = 1)
	df["Offset"] = df.apply(lambda x: x["r"]-mean_dist, axis = 1)
	y_err = np.vstack((df["mode"]-df["lower"],df["upper"]-df["mode"]))

	df["In"]   = df.apply(lambda x: ((x["r"]>=x["lower"]) and (x["r"]<=x["upper"])), axis = 1)

	print("Good distances:{0:2.1f}".format(100.*np.sum(df["In"])/len(df)))
	print("RMS: {1:0.4f}".format(prior["type"],np.sqrt(np.mean(df["Bias"]**2))))
	print("Rho: {1:0.4f}".format(prior["type"],
		np.corrcoef(df["Offset"],df["Bias"])[0,1]))

	bias[j,0] = np.sqrt(np.mean(df.loc[df["Frac"]<0.05,"Bias"]**2))
	bias[j,1] = np.sqrt(np.mean(df.loc[(df["Frac"]>0.05)&(df["Frac"]<0.1),"Bias"]**2))
	bias[j,2] = np.sqrt(np.mean(df.loc[df["Frac"]>0.1,"Bias"]**2))


	#---------- Plot fractional error ----------------------
	plt.figure(0,figsize=(6,6))
	plt.plot(mean["Frac"],mean["Diff"],lw=2,
				linestyle=prior["linestyle"],
				color=prior["color"],
				label=prior["type"])

	#---------- Plot posteriors----------------------
	plt.figure(1,figsize=(6,6))
	plt.hist(samples.flatten()-mean_dist,bins=n_bins,range=range_dist,
		histtype='step',density=True,linewidth=1,
		color=prior["color"],label=prior["type"])

	#--------- Plot bias ----------------------------
	plt.figure(2,figsize=(6,6))
	plt.errorbar(df["Offset"],df["Bias"],yerr=y_err,
		fmt='none',ls='none',marker="o",ms=5,
		ecolor="grey",elinewidth=0.01,zorder=1,label=None)
	
	plt.scatter(df["Offset"],df["Bias"],s=20,c=df["Frac"],
		marker=prior["marker"],
		zorder=2,
		label=prior["type"],
		vmax=0.1,
		cmap="viridis")


plt.figure(0)	
plt.xlabel("Fractional uncertainty")
plt.ylabel("Fractional error")
plt.xscale("log")
plt.legend(
	title="Priors",
	shadow = False,
	bbox_to_anchor=(0.0, 1.15, 1.0, 0.0),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 4,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')
pdf.savefig(bbox_inches='tight')
plt.close()

plt.figure(1)
xs = np.linspace(-100,100,1000)
plt.plot(xs,st.norm.pdf(xs,loc=0.0,scale=scl),
	linewidth=1,
	color="red",label="True")	
plt.xlabel("Offset from centre [pc]")
plt.ylabel("Density")
plt.yscale("log")
plt.ylim(1e-4,0.2)
plt.xlim(range_dist)
plt.legend(
	title="Priors",
	shadow = False,
	bbox_to_anchor=(0.0, 1.15, 1.0, 0.0),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 4,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')
pdf.savefig(bbox_inches='tight')
plt.close()

plt.figure(2)
clrb = plt.colorbar()
clrb.set_label("Fractional uncertainty")
plt.xlabel("Offset from centre [pc]")
plt.ylabel("Distance bias [pc]")
plt.ylim(-30,30)
plt.xlim(-30,30)
plt.legend(
	title="Priors",
	shadow = False,
	bbox_to_anchor=(0.0, 1.15, 1.0, 0.0),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 4,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')
pdf.savefig(bbox_inches='tight')
plt.close()
pdf.close()
#================================================================================================

#=============== Table of bias rms for prior and uncertainty regime ==========================
print(30*"-")
print("Printing table of rms bias ... ")
with open(file_tex, "w") as tex_file:
	header = ["Prior","$f_\\varpi$<0.05","0.05<$f_\\varpi$<0.1","$f_\\varpi$>0.1"]
	print("\\begin{tabular}{cccc}", file=tex_file)
	print("\\hline", file=tex_file)
	print("\\hline", file=tex_file)
	print("  &  ".join(header) + "  \\\\", file=tex_file)
	print("\\hline", file=tex_file)
	for j,prior in enumerate(list_of_priors):
		str_cols = [prior["type"]]
		for i in range(3):
			str_cols.append("{0:2.1f}".format(bias[j,i]))
		print("  &  ".join(str_cols) + "  \\\\", file=tex_file)
	print("\\hline", file=tex_file)
	print("\\end{tabular}", file=tex_file)
print("Check file: ",file_tex)




        

