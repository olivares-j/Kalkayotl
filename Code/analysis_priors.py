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


from inference import Inference
from posterior_1d import Posterior as Posterior_1d
from chain_analyser import Analysis


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines

#------------------ MCMC parameters ---------------------------------------
n_iter              = 2000         # Number of iterations for the MCMC 
n_walkers           = 5           # Number of walkers
tolerance           = 10
#------------------------------------------------------------------------------

#----------- prior parameters --------
colors      = ["blue","green","red","black"]
priors      = ["Uniform","Half-Gaussian","Half-Cauchy","EDSD"]
locations   = [0]
scales      = [1000.,1350,1500.]
styles      = ["-.","--","-"]

#============ Directories and data =================
case      = "Uniform_0_500"
dir_main   = os.getcwd()[:-4]
dir_data   = dir_main  + "Data/"
dir_ana    = dir_main  + "Analysis/"
dir_chains = dir_ana   + "Chains/"+case+"/"
dir_plots  = dir_ana   + "Plots/"
file_data  = dir_data  + case + ".csv"
file_plot  = dir_plots + case+"_error_vs_uncertainty_map.pdf"

#------- Create directories -------
if not os.path.isdir(dir_ana):
	os.mkdir(dir_ana)
if not os.path.isdir(dir_chains):
    os.mkdir(dir_chains)
if not os.path.isdir(dir_plots):
    os.mkdir(dir_plots)
#================================================

#=======================================================================================================================
data  = pn.read_csv(file_data) 
pdf = PdfPages(filename=file_plot)
plt.figure(0)
#================================== Compariosn ==========================================================================
list_observables = ["ID","parallax","parallax_error"]
for i,prior in enumerate(priors):
	print("="*30,prior,"="*30)
	for j,loc in enumerate(locations):
		print("-"*30,"Location ",loc,"-"*30)
		for k,scl in enumerate(scales):
			print(" "*30,"Scale ",scl," "*30)
			file_csv = dir_chains + "Chains_1D_"+str(prior)+"_loc="+str(int(loc))+"_scl="+str(int(scl))+".csv"

			MAP  = pn.read_csv(file_csv,usecols=["ID","dist_ctr"]) 
			df   = data.join(MAP,on="ID",lsuffix="_data",rsuffix="_MAP")

			df["Diff"] = df.apply(lambda x: (x["dist_ctr"] - x["dist"])/x["dist"], axis = 1)
			df["Frac"] = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)

			df = df.sort_values(by="Frac")

			mean = df.rolling(50).mean()
			
			#---------- Plot ----------------------
			# plt.scatter(df["Frac"],df["Diff"],s=0.1,marker=",",color=colors[i],label=None)
			plt.plot(mean["Frac"],mean["Diff"],lw=1,color=colors[i],linestyle=styles[k],label=None)


prior_lines = [mlines.Line2D([], [],color=colors[i],linestyle="-",label=prior) for i,prior in enumerate(priors)]
scl_lines   = [mlines.Line2D([], [],color="black",linestyle=styles[i],label=str(int(scl))) for i,scl in enumerate(scales)]

legend = plt.legend(handles=prior_lines,title="Priors",loc='lower left')
plt.legend(handles=scl_lines,title="Scales",loc='lower center')
plt.gca().add_artist(legend)
plt.title("Uniform at 500 pc")
plt.xlabel("Fractional uncertainty")
plt.ylabel("Fractional error")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close(0)
pdf.close()

			




        

