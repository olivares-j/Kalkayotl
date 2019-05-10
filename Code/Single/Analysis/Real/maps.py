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
import h5py
import emcee

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines


#----------- Galactic prior --------

list_of_priors = [
{"type":"EDSD",         "location":0.0,"scale":1500.0},
{"type":"Uniform",      "location":0.0,"scale":1500.0},
{"type":"Half-Cauchy",  "location":0.0,"scale":1500.0},
{"type":"Half-Gaussian","location":0.0,"scale":1500.0},
{"type":"Cauchy",       "location":300,"scale":80.0},
{"type":"Gaussian",     "location":300,"scale":80.0}
]

#============ Directories and data =================
case = "3D"
dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_ana    = dir_main  + "Analysis/NGC6774/"
dir_chains = dir_ana   + "Chains/"
file_plot  = dir_ana   + "Plots/maps_"+case+".pdf"
file_CBJ   = dir_main  + "Data/NGC6774/Olivares+2019_Bailer-Jones+208.csv"
#=======================================================================================================================

#-------- Read Coryn data ------------------------------------------------
data = pn.read_csv(file_CBJ,usecols=["ID","rest","b_rest","B_rest"])
data.sort_values(by="ID",inplace=True)
data.set_index("ID",inplace=True)

for j,prior in enumerate(list_of_priors):
	file_csv = dir_chains + "Chains_"+case+"_"+str(prior["type"])+"_loc="+str(int(prior["location"]))+"_scl="+str(int(prior["scale"]))+"_map.csv"

	infered  = pn.read_csv(file_csv,usecols=["ID","dist_min","dist_ctr","dist_max"])
	infered.rename(index=str,inplace=True, 
		columns={"dist_min": prior["type"]+"_min","dist_ctr": prior["type"]+"_ctr","dist_max": prior["type"]+"_max"})
	infered.sort_values(by="ID",inplace=True)
	infered.set_index("ID",inplace=True)
	
	data     = pn.merge(data,infered,left_index=True, right_index=True,suffixes=("_","_b"))
	
#================================== Plot points ==========================================================================

#---------- Line --------------
x_line = [250,400]

#-------- Figure ----------------------------
pdf = PdfPages(filename=file_plot)
plt.figure(figsize=(6,6))
plt.plot(x_line,x_line,color="black",linewidth=0.5,zorder=0)
for j,prior in enumerate(list_of_priors):
	x_err = np.vstack((data["rest"]-data["b_rest"],data["B_rest"]-data["rest"]))
	y_err = np.vstack((data[prior["type"]+"_ctr"]-data[prior["type"]+"_min"],data[prior["type"]+"_max"]-data[prior["type"]+"_ctr"]))
	plt.errorbar(data["rest"],data[prior["type"]+"_ctr"],yerr=y_err,xerr=x_err,label=prior["type"],#fmt='none',
		ls='none',marker="o",ms=5,elinewidth=0.01,zorder=1)
	# plt.scatter(data["rest"],data[prior["type"]+"_ctr"],s=5,zorder=2,label=prior["type"])
plt.xlabel("Distance [pc]")
plt.ylabel("Distance [pc]")
plt.xlim(np.min(data["rest"])-10,np.max(data["rest"])+10)
plt.ylim(np.min(data["rest"])-10,np.max(data["rest"])+10)
plt.legend(
	title="Prior",
	shadow = False,
	bbox_to_anchor=(0.,1.05, 1., .1),
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 3,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper left')
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()