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
from __future__ import absolute_import, print_function
import sys
import os
import numpy as np
import pandas as pn
import h5py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter

import progressbar

from adw import posterior_adw

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
fdata = "example.csv"
list_observables = ["source_id","ra","dec","parallax","ra_error","dec_error",
"parallax_error","ra_dec_corr","ra_parallax_corr","dec_parallax_corr"]

#------- reads the data and orders it
data  = pn.read_csv(fdata,usecols=list_observables,nrows=2) 
data  = data.reindex(columns=list_observables)

#------- index as string ------
data[list_observables[0]] = data[list_observables[0]].astype('str')

#------- Correct units ------
data["parallax"]       = data["parallax"]*1e-3
data["parallax_error"] = data["parallax_error"]*1e-3

data["ra_error"]       = data["ra_error"]*(1e-3 / 3600.0)
data["dec_error"]      = data["dec_error"]*(1e-3 / 3600.0)

#----- put ID as row name-----
data.set_index(list_observables[0],inplace=True)

N,D = np.shape(data)
#----------------------------------------------------
############################################################################################
labels = ["R.A. [deg]", "Dec. [deg]","Distance [pc]"]
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

#------- Open plots and H5 files 
pdf = PdfPages(filename=dir_graphs+"Distances.pdf")
fh5 = h5py.File(file_out_h5,'w')

# ------- Prepare plots --------------------
nullfmt = plt.NullFormatter()
left, width = 0.1, 0.4
bottom, height = 0.1, 0.4
bottom_h = left_h = left + width + 0.0
rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.4]
rect_histy = [left_h, bottom, 0.1, height]

#------ initialise arrays ---------
maps      = np.zeros((N,3))
medians   = np.zeros((N,3))
sds       = np.zeros((N,3))
cis       = np.zeros((N,3,2))
times     = np.zeros(N)
acc_fcs   = np.zeros(N)

#-------- Loop over stars ---------
bar = progressbar.ProgressBar(maxval=N).start()

i = 0
for ID,datum in data.iterrows():
	print(datum)

	#------ Initialise the posterior adw function -----------

	adw = posterior_adw(datum,prior=prior,prior_loc=prior_loc,prior_scale=prior_scale)
	#------- run the adw function ----------------------------
	MAP,Median,SD,CI,int_time,sample,mean_acceptance_fraction = adw.run(N_iter=N_iter)

	#------- Save sample -----
	dset = fh5.create_dataset(str(ID), data=sample)
	fh5.flush()


	#---- populate arrays----
	maps[i]     = MAP
	medians[i]  = Median
	times[i]    = int_time
	sds[i]      = SD
	cis[i,:]    = CI.T
	acc_fcs[i]  = mean_acceptance_fraction

	#---- Do plots -----------------

	for j in range(3):

		
		y_min,y_max= 0.95*np.min(sample[:,:,j]),1.05*np.max(sample[:,:,j])

		fig = plt.figure(221, figsize=(6.3, 6.3))
		ax0 = fig.add_subplot(223, position=rect_scatter)
		ax0.set_title(list_observables[0]+" : "+str(ID),pad=15)
		ax0.set_xlabel("Iteration")
		ax0.set_ylabel(labels[j])
		# ax0.set_ylim(y_min,y_max)
		ax0.plot(sample[:,:,j].T, '-', color='k', alpha=0.3,linewidth=0.3)
		ax0.axhline(MAP[j],   color='blue',ls="--",linewidth=0.5,label="MAP")
		ax0.axhline(CI[0,j], color='blue',ls=":",linewidth=0.5,label="CI 95%")
		ax0.axhline(CI[1,j], color='blue',ls=":",linewidth=0.5)
		ax0.legend(loc="upper left",ncol=4,fontsize=4)


		ax1 = fig.add_subplot(224, position=rect_histy)
		# ax1.set_ylim(y_min,y_max)
		ax1.set_ylim(ax0.get_ylim())
		ax1.axhline(MAP[j],   color='blue',ls="--",linewidth=0.5,label="MAP")
		ax1.axhline(CI[0,j],    color='blue',ls=":",linewidth=0.5,label="CI")
		ax1.axhline(CI[1,j],    color='blue',ls=":",linewidth=0.5)

		ax1.set_xlabel("Density")
		ax1.yaxis.set_ticks_position('none') 

		xticks = ax1.xaxis.get_major_ticks() 
		xticks[0].label1.set_visible(False)

		ax1.yaxis.set_major_formatter(nullfmt)
		ax1.yaxis.set_minor_formatter(nullfmt)

		ax1.hist(sample[:,:,j].flatten(),bins=100,density=True, 
			color="k",orientation='horizontal', fc='none', histtype='step',lw=0.5)
		pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
		plt.close()

	# ---- update progress bas ----
	bar.update(i+1)
	i += 1 
pdf.close()
fh5.close()

#---------- acceptance fraction ----------
print("Acceptance fraction statistics:")
print("Min.: {0:.3f}, Mean: {1:.3f}, Max.: {2:.3f}".format(np.min(acc_fcs),np.mean(acc_fcs),np.max(acc_fcs)))

#---------- return data frame----
data_out = pn.DataFrame(np.column_stack((data.index,maps,medians,sds,
	cis[:,0,0],cis[:,0,1],cis[:,1,0],cis[:,1,1],cis[:,2,0],cis[:,2,1],times)),
		columns=[list_observables[0],'map_ra','map_dec','map_parallax',
		'median_ra','median_dec','median_parallax',
		'sd_ra','sd_dec','sd_parallax',
		'ci_low_ra','ci_up_ra',
		'ci_low_dec','ci_up_dec',
		'ci_low_parallax','ci_up_parallax',
		'integrated_autocorr_time'])

data_out.to_csv(path_or_buf=file_out_csv,index=False)



        

