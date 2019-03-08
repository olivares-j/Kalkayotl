'''
Copyright 2018 Javier Olivares Romero

This file is part of Kalkayotl3D.

    Kalkayotl3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyAspidistra is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kalkayotl3D.  If not, see <http://www.gnu.org/licenses/>.
'''
#------------ LOAD LIBRARIES -------------------
from __future__ import absolute_import,division,print_function
import sys
import os
import numpy as np
import scipy.stats as st
import random
import scipy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec

import pandas as pn

from astroML.decorators import pickle_results
import progressbar
from matplotlib.ticker import NullFormatter

from adw import posterior_adw

from pygaia.astrometry.vectorastrometry import sphericalToCartesian,cartesianToSpherical



#----------------Mock data and MCMC parameters  --------------------
random_state = 1234              # Random state for the synthetic data

np.random.seed(random_state)

# ------ to test the code I use a 3d mock data distribution ---------------
data_loc               = np.array([150.0,0.0,0.0])        # Location of the mock data in parsecs
data_scale             = np.diag(np.array([10,10,10]))  # Scale, in parsecs, of the distribution for the mock data
data_distribution      = np.random.multivariate_normal  # Change it according to your needs

#-------------------------------------------------------------------------
#------- parameters of the uncertainties and correlations -----------------
#-------------------------------------------------------------------------
#   Found after fitting chi2 distribution to 100000 gaia sources
u_pax_params = [2.74,0.019,0.252]           # df=2.74,     loc=0.019,      scale=0.252
u_ra_params  = [0.9511396,0.0149999,2.1944] # df=0.9511396,loc=0.0149999,  scale=2.1944,
u_dec_params = [0.8915513,0.01369999,2.60820] # df=0.8915513,loc=0.01369999, scale=2.60820,

#   Found after fitting normal distribution to 100000 gaia sources
corr_ra_dec_params  = [0.24632236899999996,0.30250667018386196]
corr_ra_pax_params  = [-0.0952362915504452,0.2096492865445333]
corr_dec_pax_params = [-0.06340400004872164,0.17600860122703194]

#-------------------------------------------------------------------

N_samples = 1000             # Number of mock distances
N_iter    = 2000             # Number of iterations for the MCMC 

#----------- parameters for the distance prior --------
prior        = str(sys.argv[1]) #"EDSD", "Gaussian", "Uniform" or "Cauchy"
prior_loc    = int(sys.argv[2]) # Location of the prior
prior_scale  = int(sys.argv[3]) # Scale of the prior

#---------- RA and DEC priors are assumed to be uniform and independent of the distance one

#------- creates Analysis directory -------
dir_out = os.getcwd() + "/Analysis/"
if not os.path.isdir(dir_out):
	os.mkdir(dir_out)

#------ creates prior directories --------
dir_graphs = dir_out+prior+"/"+str(prior_scale)+"/"
if not os.path.isdir(dir_out+prior):
	os.mkdir(dir_out+prior)
if not os.path.isdir(dir_graphs):
	os.mkdir(dir_graphs)
#-----------------------------------

###############################################################################################

#---- calls the syn_validation function with the astroML decorator to save results
data_file = dir_graphs+"data.pkl"
@pickle_results(data_file)
def syn_validation(N_samples,data_loc,data_scale,random_state=random_state):
	print("Generating synthetic data ...")
	#---------- create synthetic data -------------------------------------------------------------------------
	true_xyz = data_distribution(mean=data_loc, cov=data_scale, size=N_samples)
     
	#---- Transform XYZ to RA,DEC,parallax -------
	true_dst,ra,dec = cartesianToSpherical(true_xyz[:,0],true_xyz[:,1],true_xyz[:,2])

	ra  = np.degrees(ra)  # transform to degrees
	dec = np.degrees(dec) # transform to degrees
	pax = 1.0/true_dst    # Transform to parallax

	#----- assigns uncertainties and correlations similar to those present in Gaia data ------- 
	u_pax    = st.chi2.rvs(df=u_pax_params[0],loc=u_pax_params[1],scale=u_pax_params[2],size=N_samples,random_state=random_state) #Units in mas
	u_ra     = st.chi2.rvs(df=u_ra_params[0], loc=u_ra_params[1], scale=u_ra_params[2], size=N_samples,random_state=random_state) #Units in mas
	u_dec    = st.chi2.rvs(df=u_dec_params[0],loc=u_dec_params[1],scale=u_dec_params[2],size=N_samples,random_state=random_state) #Units in mas

	#------ Transform uncertainties to degrees and arcseconds --------
	u_pax    = u_pax/1000.0
	u_ra     = u_ra/3.6e6
	u_dec    = u_dec/3.6e6


	corr_ra_dec  = st.norm.rvs(corr_ra_dec_params[0], corr_ra_dec_params[1],  size=N_samples,random_state=random_state)
	corr_ra_pax  = st.norm.rvs(corr_ra_pax_params[0], corr_ra_pax_params[1],  size=N_samples,random_state=random_state)
	corr_dec_pax = st.norm.rvs(corr_dec_pax_params[0],corr_dec_pax_params[1], size=N_samples,random_state=random_state)

	#----------------- array of data -------------------------
	data = np.column_stack((ra,dec,pax,u_ra,u_dec,u_pax,corr_ra_dec,corr_ra_pax,corr_dec_pax))
	#----------------------------------------------------

	#------------------------------------------------------------
	print("Checking for singular matrices")
	for i,datum in enumerate(data):
		singular = True
		count    = 0
		#------ Run loop until matrix is not singular
		while singular :
			count += 1
			corr      = np.zeros((3,3))
			corr[0,1] = datum[6]
			corr[0,2] = datum[7]
			corr[1,2] = datum[8]
			corr      = np.eye(3) + corr + corr.T 
			cov       = np.diag(datum[3:6]).dot(corr.dot(np.diag(datum[3:6])))

			try:
				s,logdet = np.linalg.slogdet(cov)
			except Exception as e:
				print(e)

			if s <= 0:
				print("Replacing singular matrix")
				#------- If matrix is singular then replace the values
				datum[6]  = st.norm.rvs(corr_ra_dec_params[0], corr_ra_dec_params[1], size=1,random_state=random_state+count)[0]
				datum[7]  = st.norm.rvs(corr_ra_pax_params[0], corr_ra_pax_params[1], size=1,random_state=random_state+count)[0]
				datum[8]  = st.norm.rvs(corr_dec_pax_params[0],corr_dec_pax_params[1],size=1,random_state=random_state+count)[0]
			else:
				singular  = False
				data[i,:] = datum
	#--------------------------

	# ------- Prepare plots --------------------
	pdf = PdfPages(filename=dir_graphs+"Sample_of_distances.pdf")
	random_sample = np.random.randint(0,N_samples,size=10) #Only ten plots

	nullfmt = plt.NullFormatter()
	left, width = 0.1, 0.4
	bottom, height = 0.1, 0.4
	bottom_h = left_h = left + width + 0.0
	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.4]
	rect_histy = [left_h, bottom, 0.1, height]

	frac_error= np.zeros_like(true_xyz)
	maps      = np.zeros_like(true_xyz)
	times     = np.zeros(N_samples)
	sds       = np.zeros_like(true_xyz)
	cis       = np.zeros((N_samples,2,3))
	acc_fcs   = np.zeros(N_samples)

	print("Sampling the posteriors ...")

	bar = progressbar.ProgressBar(maxval=N_samples).start()
	
	for d,(datum,t_xyz,t_dst) in enumerate(zip(data,true_xyz,true_dst)):
		###################### Initialise the posterio_adw class ################################

		adw = posterior_adw(datum,prior=prior,prior_loc=prior_loc,prior_scale=prior_scale)
		#------- run the p2d function ----------------------------
		MAP,Median,SD,CI,int_time,sample,mean_acceptance_fraction = adw.run(N_iter=N_iter)

		obs_xyz = sphericalToCartesian(MAP[2],MAP[0],MAP[1])

		
		#---- populate arrays----
		frac_error[d,0]  = (MAP[2] - t_dst)/t_dst
		frac_error[d,1]  = (MAP[0] - datum[0])/datum[0]
		frac_error[d,2]  = (MAP[1] - datum[1])/datum[1]
		maps[d,:]      = MAP
		times[d]       = int_time
		sds[d,:]       = SD
		cis[d,:,:]     = CI
		acc_fcs[d]     = mean_acceptance_fraction

		#---- plot just random sample

		if d in random_sample:
			y_min,y_max= 0.95*np.min(sample[:,:,2]),1.05*np.max(sample[:,:,2])

			fig = plt.figure(221, figsize=(6.3, 6.3))
			ax0 = fig.add_subplot(223, position=rect_scatter)
			ax0.set_xlabel("Iteration")
			ax0.set_ylabel("Distance [pc]")
			ax0.set_ylim(y_min,y_max)
			ax0.plot(sample[:,:,2].T, '-', color='k', alpha=0.3,linewidth=0.3)
			ax0.axhline(t_dst,  color='black',ls="solid",linewidth=0.8,label="True")
			ax0.axhline(MAP[2],   color='blue',ls="--",linewidth=0.5,label="MAP")
			ax0.axhline(CI[0,2], color='blue',ls=":",linewidth=0.5,label="CI 95%")
			ax0.axhline(CI[1,2], color='blue',ls=":",linewidth=0.5)
			ax0.legend(loc="upper left",ncol=4,fontsize=4)


			ax1 = fig.add_subplot(224, position=rect_histy)
			ax1.set_ylim(y_min,y_max)
			ax1.axhline(t_dst,  color='black',ls="solid",linewidth=0.8,label="True")
			ax1.axhline(MAP[2],   color='blue',ls="--",linewidth=0.5,label="MAP")
			ax1.axhline(CI[0,2],    color='blue',ls=":",linewidth=0.5,label="CI")
			ax1.axhline(CI[1,2],    color='blue',ls=":",linewidth=0.5)

			ax1.set_xlabel("Density")
			ax1.yaxis.set_ticks_position('none') 

			xticks = ax1.xaxis.get_major_ticks() 
			xticks[0].label1.set_visible(False)

			ax1.yaxis.set_major_formatter(nullfmt)
			ax1.yaxis.set_minor_formatter(nullfmt)

			ax1.hist(sample[:,:,2].flatten(),bins=100,density=True, 
				color="k",orientation='horizontal', fc='none', histtype='step',lw=0.5)
			pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
			plt.close()

		# ---- update progress bas ----
		bar.update(d) 
	pdf.close()

	print("Acceptance fraction statistics:")
	print("Min.: {0:.3f}, Mean: {1:.3f}, Max.: {2:.3f}".format(np.min(acc_fcs),np.mean(acc_fcs),np.max(acc_fcs)))

	#---------- return data frame----
	data = pn.DataFrame(np.column_stack((ra,dec,true_dst,pax,u_ra,u_dec,u_pax,u_pax/pax,u_ra/ra,u_dec/dec,maps,frac_error,sds,times)),
		columns=['True RA','True DEC','True distance','True parallax','RA uncertainty','DEC uncertainty','Parallax uncertainty',
				 'Parallax fractional uncertainty','RA fractional uncertainty','DEC fractional uncertainty',
		         'MAP RA','MAP DEC','MAP distance',
		         'Distance fractional error','RA fractional error','DEC fractional error',
		         'SD RA','SD DEC','SD distance','Autocorrelation time'])
	return data

data = syn_validation(N_samples,data_loc=data_loc,data_scale=data_scale)

data.sort_values('Parallax fractional uncertainty',inplace=True)

nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.7
bottom, height = 0.1, 0.7
bottom_h = left_h = left + width + 0.01

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

pdf = PdfPages(filename=dir_graphs+"Errors.pdf")
##################### DISTANCE ERRORS ##################################
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)
axHistx   = plt.axes(rect_histx)
axHisty   = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

asymetric_uncert = [data['SD distance']/data['True distance'],data['SD distance']/data['True distance']]

# the scatter plot:
axScatter.errorbar(x=data["Parallax fractional uncertainty"],y=data["Distance fractional error"],yerr=asymetric_uncert,
	elinewidth=0.1,alpha=0.1,fmt=".",ecolor="grey",color="blue",ms=0.1)
axScatter.scatter(x=data["Parallax fractional uncertainty"],y=data["Distance fractional error"],color="blue",s=0.5)
axScatter.set_xlabel("Parallax fractional uncertainty")
axScatter.set_ylabel("Distance fractional error")

axHistx.hist(data["Parallax fractional uncertainty"], color="blue",bins=100,density=True)
axHisty.hist(data["Distance fractional error"], color="blue",bins=100,density=True,orientation="horizontal")

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

plt.text(1.2,0.9,s="MAD "+'{0:.3f}'.format(np.mean(np.abs(data["Distance fractional error"]))),fontsize=8,
	horizontalalignment='center',verticalalignment='center', transform=axHistx.transAxes)
plt.text(1.2,0.7,s="Mean "+'{0:.3f}'.format(np.mean(data["Distance fractional error"])),fontsize=8,
	horizontalalignment='center',verticalalignment='center', transform=axHistx.transAxes)
plt.text(1.2,0.5,s="SD "+'{0:.3f}'.format(np.std(data["Distance fractional error"])),fontsize=8,
	horizontalalignment='center',verticalalignment='center', transform=axHistx.transAxes)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
##########################################################################################
##################### RA error ###############################
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)
axHistx   = plt.axes(rect_histx)
axHisty   = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

asymetric_uncert = [data['SD RA']/data['True RA'],data['SD RA']/data['True RA']]

# the scatter plot:
axScatter.errorbar(x=data["RA fractional uncertainty"],y=data["RA fractional error"],yerr=asymetric_uncert,
	elinewidth=0.1,alpha=0.1,fmt=".",ecolor="grey",color="blue",ms=0.1)
axScatter.scatter(x=data["RA fractional uncertainty"],y=data["RA fractional error"],color="blue",s=0.5)
axScatter.set_xlabel("RA fractional uncertainty")
axScatter.set_ylabel("RA fractional error")

axHistx.hist(data["RA fractional uncertainty"], color="blue",bins=100,density=True)
axHisty.hist(data["RA fractional error"], color="blue",bins=100,density=True,orientation="horizontal")

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

plt.text(1.2,0.9,s="MAD "+'{0:.3f}'.format(np.mean(np.abs(data["RA fractional error"]))),fontsize=8,
	horizontalalignment='center',verticalalignment='center', transform=axHistx.transAxes)
plt.text(1.2,0.7,s="Mean "+'{0:.3f}'.format(np.mean(data["RA fractional error"])),fontsize=8,
	horizontalalignment='center',verticalalignment='center', transform=axHistx.transAxes)
plt.text(1.2,0.5,s="SD "+'{0:.3f}'.format(np.std(data["RA fractional error"])),fontsize=8,
	horizontalalignment='center',verticalalignment='center', transform=axHistx.transAxes)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()

#----- Plots the distribution of integrated autocorrelation times ------------
pdf = PdfPages(filename=dir_graphs+"AutoTimes.pdf")
plt.hist(data["Autocorrelation time"], bins=100)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()



        

