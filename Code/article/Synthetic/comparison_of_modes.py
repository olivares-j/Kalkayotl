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
{"type":"EDSD",     "marker":"."},
# {"type":"Uniform",  "marker":"v"},
# {"type":"Gaussian", "marker":"s"},
# {"type":"Cauchy",   "marker":"d"}
]

#============ Directories and data =================
dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_out    = dir_main  + "Outputs/"
dir_chains = dir_out   + case + "/"
file_plot  = dir_out   + "Plots/Mode_comparison_" + case + ".pdf"
file_data  = dir_main  + "Data/" + case + "/"  + case + ".csv"
#=======================================================================================================================

#-------- Read data ------------------------------------------------
data = pn.read_csv(file_data,usecols=["ID","r","parallax","parallax_error"])
data.sort_values(by="ID",inplace=True)
data.set_index("ID",inplace=True)


#--------------- Loop over priors ---------------------------------------------------------------------------
for j,prior in enumerate(list_of_priors):
	#------ Read modes --------------------------------------------------------
	file_csv = dir_chains + prior["type"]+"/1D/Sources_mode.csv"
	infered  = pn.read_csv(file_csv,usecols=["ID","mode","lower","upper"])
	infered.rename(index=str,inplace=True, 
		columns={"lower": prior["type"]+"_min","mode": prior["type"]+"_ctr","upper": prior["type"]+"_max"})
	infered.sort_values(by="ID",inplace=True)
	infered.set_index("ID",inplace=True)

	#------------ Merge with data ---------------------------------------------------------
	data     = pn.merge(data,infered,left_index=True, right_index=True,suffixes=("_","_b"))

#================================== Plot points ==========================================================================

#---------- Line --------------
x_line = [250,400]

#-------- Figure ----------------------------
pdf = PdfPages(filename=file_plot)
plt.figure(figsize=(6,6))
plt.plot(x_line,x_line,color="black",linewidth=0.5,zorder=0)
for j,prior in enumerate(list_of_priors):
	x     = data["r"]
	y     = data[prior["type"]+"_ctr"]
	clr   = data["parallax_error"]/data["parallax"]
	y_err = np.vstack((data[prior["type"]+"_ctr"]-data[prior["type"]+"_min"],
		               data[prior["type"]+"_max"]-data[prior["type"]+"_ctr"]))
	
	plt.errorbar(x,y,yerr=y_err,
		fmt='none',ls='none',marker="o",ms=5,
		ecolor="grey",elinewidth=0.01,zorder=1,label=None)
	
	plt.scatter(x,y,s=20,c=clr,marker=prior["marker"],zorder=2,label=prior["type"],cmap="magma")

plt.xlabel("True distance [pc]")
plt.ylabel("Kalkayotl distance [pc]")
plt.colorbar()
plt.xlim(np.min(data["r"])-10,np.max(data["r"])+10)
plt.ylim(np.min(data["r"])-10,np.max(data["r"])+10)
plt.legend(
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
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()