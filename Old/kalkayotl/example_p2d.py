'''
Copyright 2018 Javier Olivares Romero

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

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec

import scipy.stats as st
import random

import pandas as pn
import h5py

# import progressbar
from matplotlib.ticker import NullFormatter

from p2d import parallax2distance

#----------------MCMC parameters  --------------------

N_iter    = 2000              # Number of iterations for the MCMC 

#----------- prior parameters --------
prior        = str(sys.argv[1]) #"EDSD", "Gaussian", "Uniform" or "Cauchy"
prior_loc    = int(sys.argv[2]) # Location of the prior
prior_scale  = int(sys.argv[3]) # Scale of the prior


################################ DATA SET ##################################################
# Reds the data set.
# Keep the order of observables, uncertainties and correlations
# IMPORTANT put the identifier first
fdata = "/home/javier/Repositories/Kalkayotl/Data/Star_300_0.csv"
list_observables = ["ID","parallax","parallax_error"]

#------- reads the data and orders it
data  = pn.read_csv(fdata,usecols=list_observables,na_values=99.0) 
#---------- drop na values ------------
data  = data.dropna(thresh=2)
data  = data.reindex(columns=list_observables)

#------- index as string ------
data[list_observables[0]] = data[list_observables[0]].astype('str')

#------- Correct units ------
data["parallax"]       = data["parallax"]*1e-3
data["parallax_error"] = data["parallax_error"]*1e-3

#----- put ID as row name-----
data.set_index(list_observables[0],inplace=True)

N_samples,D = np.shape(data)
#----------------------------------------------------
############################################################################################

#------- creates Analysis directory -------
dir_out = os.getcwd() + "/Example/"
if not os.path.isdir(dir_out):
	os.mkdir(dir_out)

#------ creates prior directories --------
dir_graphs = dir_out+prior+"/"+str(prior_scale)+"/"
if not os.path.isdir(dir_out+prior):
	os.mkdir(dir_out+prior)
if not os.path.isdir(dir_graphs):
	os.mkdir(dir_graphs)
#-----------------------------------

file_out_csv= dir_graphs + "out_"+str(prior)+"_"+str(prior_loc)+"_"+str(prior_scale)+".csv"        # file where the statistics will be written.
file_out_h5 = dir_graphs + "out_"+str(prior)+"_"+str(prior_loc)+"_"+str(prior_scale)+".hdf5"
###################### Initialise the parallax2distance class ################################

p2d = parallax2distance(N_iter=N_iter,nwalkers=50,prior=prior,prior_loc=prior_loc,prior_scale=prior_scale,quantiles=[2.3,97.7])

###############################################################################################

#------- Open plots and H5 files 
pdf = PdfPages(filename=dir_graphs+"Distances.pdf")
fh5 = h5py.File(file_out_h5,'w')

nullfmt = plt.NullFormatter()
left, width = 0.1, 0.4
bottom, height = 0.1, 0.4
bottom_h = left_h = left + width + 0.0
rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.4]
rect_histy = [left_h, bottom, 0.1, height]

#------ initialise arrays ---------
maps      = np.zeros(N_samples)
medians   = np.zeros(N_samples)
sds       = np.zeros(N_samples)
cis       = np.zeros((N_samples,2))
times     = np.zeros(N_samples)

# bar = progressbar.ProgressBar(maxval=N_samples).start()

i=0
for ID,datum in data.iterrows():
	#------- run the p2d function ----------------------------
	MAP,Median,SD,CI,int_time,sample = p2d.run(datum["parallax"],datum["parallax_error"])

	#------- Save sample -----
	dset = fh5.create_dataset(str(ID), data=sample)
	fh5.flush()

	#---- populate arrays----
	maps[i]     = MAP
	medians[i]  = Median
	times[i]    = int_time
	sds[i]      = SD
	cis[i,:]    = CI

	#---- plot posterior distance sample


	y_min,y_max= 0.95*np.min(sample),1.05*np.max(sample)

	fig = plt.figure(221, figsize=(6.3, 6.3))
	ax0 = fig.add_subplot(223, position=rect_scatter)
	ax0.set_title(list_observables[0]+" : "+str(ID),pad=15)
	ax0.set_xlabel("Iteration")
	ax0.set_ylabel("Distance [pc]")
	ax0.set_ylim(y_min,y_max)
	ax0.plot(sample.T, '-', color='k', alpha=0.3,linewidth=0.3)
	ax0.axhline(MAP,   color='blue',ls="--",linewidth=0.5,label="MAP")
	ax0.axhline(CI[0], color='blue',ls=":",linewidth=0.5,label="CI")
	ax0.axhline(CI[1], color='blue',ls=":",linewidth=0.5)
	ax0.legend(loc="upper left",ncol=4,fontsize=4)


	ax1 = fig.add_subplot(224, position=rect_histy)
	ax1.set_ylim(y_min,y_max)
	ax1.axhline(MAP,   color='blue',ls="--",linewidth=0.5,label="MAP")
	ax1.axhline(CI[0],    color='blue',ls=":",linewidth=0.5,label="CI")
	ax1.axhline(CI[1],    color='blue',ls=":",linewidth=0.5)

	ax1.set_xlabel("Density")
	ax1.yaxis.set_ticks_position('none') 

	xticks = ax1.xaxis.get_major_ticks() 
	xticks[0].label1.set_visible(False)

	ax1.yaxis.set_major_formatter(nullfmt)
	ax1.yaxis.set_minor_formatter(nullfmt)

	ax1.hist(sample.flatten(),bins=100,density=True, 
		color="k",orientation='horizontal', fc='none', histtype='step',lw=0.5)
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()

	# ---- update progress bas ----
	# bar.update(i+1)
	i += 1 
pdf.close()
fh5.close()

#---------- return data frame----
data_out = pn.DataFrame(np.column_stack((data.index,maps,medians,sds,
	cis[:,0],cis[:,1],times)),
		columns=[list_observables[0],'map_distance',
		'median_distance',
		'sd_distance',
		'ci_low_distance','ci_up_distance',
		'integrated_autocorr_time'])

data_out.to_csv(path_or_buf=file_out_csv,index=False)



        

