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
prior = "EDSD"

#============ Directories and data =================
dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_out    = dir_main  + "Outputs/"
dir_chains = dir_out   + case + "/"
file_plot  = dir_out   + "Plots/Correlation_comparison_" + case + ".pdf"
#=======================================================================================================================


#------ Correlated --------------------------------------------------------
file_corr   = dir_chains + prior+"/1D_correlated/Sources_mode.csv"
df_corr     = pn.read_csv(file_corr,usecols=["ID","mode","lower","upper"])
df_corr.rename(index=str,inplace=True, 
	columns={"lower": "corr_min","mode":"corr_ctr","upper": "corr_max"})
df_corr.sort_values(by="ID",inplace=True)
df_corr.set_index("ID",inplace=True)

#------ Uncorrelated --------------------------------------------------------
file_uncorr = dir_chains + prior+"/1D_uncorrelated/Sources_mode.csv"
df_uncorr   = pn.read_csv(file_uncorr,usecols=["ID","mode","lower","upper"])
df_uncorr.rename(index=str,inplace=True, 
	columns={"lower": "uncorr_min","mode":"uncorr_ctr","upper": "uncorr_max"})
df_uncorr.sort_values(by="ID",inplace=True)
df_uncorr.set_index("ID",inplace=True)

#------------ Merge with data ---------------------------------------------------------
data     = pn.merge(df_corr,df_uncorr,left_index=True, right_index=True,suffixes=("_","_b"))

#================================== Plot points ==========================================================================

#---------- Line --------------
x_line = [250,400]

#-------- Figure ----------------------------
pdf = PdfPages(filename=file_plot)
plt.figure(figsize=(6,6))
plt.plot(x_line,x_line,color="black",linewidth=0.5,zorder=0)
for j,prior in enumerate(list_of_priors):
	x     = data["corr_ctr"]
	y     = data["uncorr_ctr"]
	x_err = np.vstack((data["corr_ctr"]-data["corr_min"],
		               data["corr_max"]-data["corr_ctr"]))
	y_err = np.vstack((data["uncorr_ctr"]-data["uncorr_min"],
		               data["uncorr_max"]-data["uncorr_ctr"]))
	
	plt.errorbar(x,y,yerr=y_err,xerr=x_err,
		fmt='none',ls='none',marker="o",ms=5,
		ecolor="grey",elinewidth=0.01,zorder=1,label=None)
	
	plt.scatter(x,y,s=20,zorder=2)

plt.xlabel("Correlated distance [pc]")
plt.ylabel("Uncorrelated distance [pc]")
plt.xlim(np.min(data["corr_ctr"])-10,np.max(data["corr_ctr"])+10)
plt.ylim(np.min(data["uncorr_ctr"])-10,np.max(data["uncorr_ctr"])+10)
# plt.legend(
# 	title="Prior",
# 	shadow = False,
# 	bbox_to_anchor=(0.,1.01, 1., .1),
# 	borderaxespad=0.,
# 	frameon = True,
# 	fancybox = True,
# 	ncol = 4,
# 	fontsize = 'smaller',
# 	mode = 'expand',
# 	loc = 'upper left')
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()