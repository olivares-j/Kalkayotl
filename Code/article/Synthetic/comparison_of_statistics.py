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

#===================== Knobs ====================================
#----------- prior --------
prior = "Gaussian"

#-------- Summary statistics ----------------------------------------
list_of_statistics =[
{"type":"mode",  "color":"blue","line_style":"-",   "name":"Mode"},
{"type":"median","color":"green","line_style":"-",  "name":"Median"},
{"type":"mean",  "color":"orange","line_style":"-", "name":"Mean"},
]
#---------------------------------------------------------------------

#-----------------Directories and data -----------------------
case       = "Gauss_1000_1e3"
dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_data   = dir_main  + "Data/Synthetic/"
dir_out    = dir_main  + "Outputs/Synthetic/"
dir_plots  = dir_main  + "Outputs/Plots/"
dir_chains = dir_out   + case + "/" + prior + "/" + "1D/"
file_data  = dir_data  + case + "/" +case+".csv"
file_plot  = dir_plots + "Comparison_of_statistics.pdf"
#====================================================================

#=========== Data and directories ==================
#------- Create directories -------
os.makedirs(dir_plots,exist_ok=True)
#-----------------------------------

#-------- Load data -----------
data  = pn.read_csv(file_data,usecols=["ID","r","parallax","parallax_error"]) 
#---------------------------------
#================================================

#================================== Plots ==========================================================================
pdf = PdfPages(filename=file_plot)
plt.figure(0)

#-------- Loop over summary statistics -------------------------------------------
for statistic in list_of_statistics:
	#------- Load summary ------------------------------------
	file_csv = dir_chains + "Sources_"+statistic["type"]+".csv"
	infered  = pn.read_csv(file_csv,usecols=["ID",statistic["type"]])

	#-------- Cross-identify --------------------------------------------------
	df       = data.join(infered,on="ID",lsuffix="_data",rsuffix="_chain")

	#----------- Compute frational error and uncertainty -----------------------------
	df["Diff"] = df.apply(lambda x: abs(x[statistic["type"]] - x["r"])/x["r"], axis = 1)
	df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
	df = df.sort_values(by="Frac")

	#--- Estimate the rolling mean ---
	mean = df.rolling(5).mean()

	#---------- Plot ----------------------
	plt.plot(mean["Frac"],mean["Diff"],linestyle=statistic["line_style"],color=statistic["color"],label=statistic["name"])

plt.ylabel("Fractional error")
plt.xlabel("Fractional uncertainty")
plt.xscale("log")
plt.yscale("log")
plt.xlim(0.01,1.0)
plt.ylim(1e-3,0.01)
plt.legend(
	shadow = False,
	bbox_to_anchor=(0.0, 1.07, 1., .0),
    borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 3,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper center')
pdf.savefig(bbox_inches='tight')
plt.close(0)
pdf.close()
#================================================================================================================================

			




        

