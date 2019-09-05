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

case = "Gauss_1"
#----------- prior --------
list_of_priors = [
	{"type":"EDSD",          "marker":"."},
	# {"type":"Half-Cauchy",   "marker":"s"},
	# {"type":"Half-Gaussian", "marker":"d"},
	{"type":"Uniform",       "marker":"+"},
	{"type":"Cauchy",        "marker":"p"},
	{"type":"Gaussian",      "marker":"v"},
	# {"type":"GMM",           "marker":"x"}
	]

#============ Directories and data =================
dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_out    = dir_main  + "Outputs/"
dir_chains = dir_out   + case + "/"
file_plot  = dir_out   + "Plots/Comparison_of_correlations.pdf"
file_data  = dir_main  + "Data/" + case + "/" + case +".csv"
#=======================================================================================================================

#------- Read Data ------------------------------
data = pn.read_csv(file_data,usecols=["ID","r"])
data.sort_values(by="ID",inplace=True)
data.set_index("ID",inplace=True)

#======================= Plots =================================================
#---------- Line --------------
x_line = [200,400]
y_lims = [280,380]

#-------- Figure ----------------------------
pdf = PdfPages(filename=file_plot)
fig, axs = plt.subplots(2, 1, sharex=True,figsize=(6,12))

for prior in list_of_priors:
	#------ Uncorrelated --------------------------------------------------------
	file_uncorr = dir_chains + prior["type"]+"/1D_uncorrelated/Sources_mode.csv"
	df_uncorr   = pn.read_csv(file_uncorr,usecols=["ID","mode","lower","upper"])
	df_uncorr.rename(index=str,inplace=True, 
		columns={"lower": "uncorr_min","mode":"uncorr_ctr","upper": "uncorr_max"})
	df_uncorr.sort_values(by="ID",inplace=True)
	df_uncorr.set_index("ID",inplace=True)

	#------------ Merge with data ---------------------------------------------------------
	df     = pn.merge(data,df_uncorr,left_index=True, right_index=True,suffixes=("_","_b"))

	#------ Correlated --------------------------------------------------------
	file_corr   = dir_chains + prior["type"]+"/1D_correlated/Sources_mode.csv"
	df_corr     = pn.read_csv(file_corr,usecols=["ID","mode","lower","upper"])
	df_corr.rename(index=str,inplace=True, 
		columns={"lower": "corr_min","mode":"corr_ctr","upper": "corr_max"})
	df_corr.sort_values(by="ID",inplace=True)
	df_corr.set_index("ID",inplace=True)

	#------------ Merge with data ---------------------------------------------------------
	df     = pn.merge(df,df_corr,left_index=True, right_index=True,suffixes=("_","_b"))


	
	x        = df["r"]
	y_corr   = df["corr_ctr"]
	y_uncorr = df["uncorr_ctr"]

	axs[0].plot(x_line,x_line,color="black",linewidth=0.5,zorder=0)
	axs[1].plot(x_line,x_line,color="black",linewidth=0.5,zorder=0)


	axs[0].scatter(x,y_uncorr,s=5,zorder=2,marker=prior["marker"],label=prior["type"])
	axs[1].scatter(x,y_corr,s=5,zorder=2,marker=prior["marker"],label=prior["type"])

axs[0].set_ylabel("Uncorrelated distance [pc]")
axs[1].set_xlabel("True distance [pc]")
axs[1].set_ylabel("Correlated distance [pc]")
axs[0].set_ylim(y_lims)
axs[0].set_xlim(280,340)
axs[1].set_xlim(280,340)
axs[1].set_ylim(y_lims)

axs[0].legend(
	title="Prior",
	shadow = False,
	bbox_to_anchor=(0.,1.01, 1., .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 4,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')

plt.subplots_adjust(wspace=0, hspace=0.05)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()

pdf.close()