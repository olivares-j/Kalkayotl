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

#----------------Mock data and MCMC parameters  --------------------
random_state        = 1234              # Random state for the synthetic data

data_loc,data_scale = 300, 20   # Location and scale of the distribution for the mock data
data_distribution   = st.norm # Change it according to your needs
n_stars             = 1000         # Number of mock distances
labels              = ["ID","dist","parallax","parallax_error"]
n_iter              = 3000         # Number of iterations for the MCMC 
n_walkers           = 10           # Number of walkers
#------------------------------------------------------------------------------

#----------- prior parameters --------
colors      = ["blue","green","red","black"]
priors      = ["Uniform","Half-Gaussian","Half-Cauchy","EDSD"]
locations   = [0]
scales      = [500.,1000.,1350,1500.]

#============ Directories and data =================
dir_main  = os.getcwd()[:-4]
dir_data  = dir_main + "Data/"
dir_expl  = dir_main + "Analysis/"
dir_chains= dir_expl + "Chains/"
dir_plots = dir_expl + "Plots/"
file_data = dir_data + "synthetic.csv"

#------- Create directories -------
if not os.path.isdir(dir_expl):
	os.mkdir(dir_expl)
if not os.path.isdir(dir_chains):
    os.mkdir(dir_chains)
if not os.path.isdir(dir_plots):
    os.mkdir(dir_plots)
#================================================


#====================== Generate Synthetic Data ============================================================================
if not os.path.isfile(file_data):
	#---------- create synthetic data -------------------------------------------------------------------------
	dst      = data_distribution.rvs(loc=data_loc, scale=data_scale, size=n_stars,random_state=random_state)
	#---- obtains the parallax -------
	pax      = 1.0/dst
	#----- assigns an uncertainty similar to those present in TGAS data ------- 
	u_pax    = st.chi2.rvs(df=2.54,loc=0.21e-3, scale=0.069e-3, size=n_stars,random_state=random_state) #Values from fit to TGAS data
	#----------------------------------------------------
	data = np.column_stack((dst,pax*1e3,u_pax*1e3))
	df = pn.DataFrame(data=data,columns=labels[1:])
	df.to_csv(path_or_buf=file_data,index_label=labels[0])
#=======================================================================================================================
data  = pn.read_csv(file_data) 
pdf = PdfPages(filename=dir_plots+"Errors_vs_uncertainty.pdf")
plt.figure()
#================================== Inference ==========================================================================

list_observables = ["ID","parallax","parallax_error"]
for i,prior in enumerate(priors):
	print("="*30,prior,"="*30)
	for loc in locations:
		print("-"*30,"Location ",loc,"-"*30)
		for scl in scales:
			print(" "*30,"Scale ",scl," "*30)
			name_chains = "Chains_1D_"+str(prior)+"_loc="+str(int(loc))+"_scl="+str(int(scl))+".h5"
			file_chains = dir_chains+name_chains

			if not os.path.isfile(file_chains):
				p1d = Inference(posterior=Posterior_1d,
				                prior=prior,
				                prior_loc=loc,
				                prior_scale=scl,
				                n_walkers=n_walkers)
				p1d.load_data(file_data,list_observables)
				p1d.run(n_iter,file_chains=file_chains,tol_convergence=20)

			a1d = Analysis(n_dim=1,file_name=file_chains,id_name=labels[0],dir_plots=dir_plots)
			# a1d.plot_chains(names="1")
			MAP = a1d.get_MAP()

			diff = (MAP[0] - data[labels[1]])/data[labels[1]]
			frac = data[labels[3]]/data[labels[2]]

			idx  = np.argsort(frac)
			frac = frac[idx]
			diff = diff[idx]

			df   = pn.DataFrame(data=np.column_stack((frac,diff)),columns=["Frac","Diff"])
			mean = df.rolling(3).mean()
			
			#---------- Plot ----------------------
			plt.scatter(frac,diff,s=0.1,marker=",",color=colors[i],label=None)
			plt.plot(mean["Frac"],mean["Diff"],lw=0.1,color=colors[i],label=prior+" "+str(int(scl)))
plt.legend()
plt.title("Cluster at 300 pc")
plt.xlabel("Fractional uncertainty")
plt.ylabel("Fractional error")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()

			




        

