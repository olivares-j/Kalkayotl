#! /opt/anaconda2-4.3.1/envs/idp/bin/python
import sys
import os
import numpy as np
import scipy.stats as st
import random

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec

import pandas as pn

from astroML.decorators import pickle_results
import progressbar
from matplotlib.ticker import NullFormatter

from p2d import parallax2distance

dir_    = os.getcwd()
dir_out = dir_ + "/Analysis/"
os.mkdir(dir_out)

#---------------- Reads the data --------------------
random_state = 1234              # Random state fo rht synthetic data

data_loc,data_scale    = 0,500   # Location and scale of the distribution for the mock data

N_samples = 10                 # Number of mock distances
N_iter    = 2000                 # Number of iterations for the MCMC 

prior        = str(sys.argv[1]) #"EDSD", "Gaussian", "Uniform" o "Cauchy"
prior_loc    = int(sys.argv[2]) # Location of the prior
prior_scale  = int(sys.argv[3]) # Scale of the prior

#------ creates directories --------
dir_graphs = dir_out+prior+"/"+str(prior_scale)+"/"
if not os.path.isdir(dir_out+prior):
	os.mkdir(dir_out+prior)
if not os.path.isdir(dir_graphs):
	os.mkdir(dir_graphs)
#-----------------------------------

data_file = dir_graphs+"data.pkl"

p2d = parallax2distance(N_iter=N_iter,prior=prior,prior_loc=prior_loc,prior_scale=prior_scale)


@pickle_results(data_file)
def syn_validation(N_samples,data_loc,data_scale,random_state=1234):
	#---------- careate synthetic data --------
	# true_dst = st.norm.rvs(loc=data_loc, scale=data_scale, size=N_samples,random_state=random_state)
	true_dst = st.uniform.rvs(loc=data_loc, scale=data_scale, size=N_samples,random_state=random_state)
	pax      = map(lambda x: 1/x, true_dst)
	u_pax    = st.chi2.rvs(df=2.54,loc=0.21e-3, scale=0.069e-3, size=N_samples,random_state=random_state) #Values from fit to TGAS data
	ru_pax   = u_pax/pax
	#----------------------------------------------------

	# ------- Preapare plots --------------------
	pdf = PdfPages(filename=dir_graphs+"Sample_of_distances.pdf")
	random_sample = np.random.randint(0,N_samples,size=10) #Only ten plots

	nullfmt = plt.NullFormatter()
	left, width = 0.1, 0.4
	bottom, height = 0.1, 0.4
	bottom_h = left_h = left + width + 0.0
	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.4]
	rect_histy = [left_h, bottom, 0.1, height]

	rel_error = np.ones_like(true_dst)
	maps      = np.zeros_like(true_dst)
	times     = np.zeros_like(true_dst)
	sds       = np.zeros_like(true_dst)
	cis       = np.zeros((N_samples,2))

	bar = progressbar.ProgressBar(maxval=N_samples).start()
	
	for d,(plx,u_plx,tdst) in enumerate(zip(pax,u_pax,true_dst)):
		#------- run the p2d function ----------------------------
		MAP,Mean,SD,CI,int_time,sample = p2d.run(plx,u_plx)
		
		#---- populate arrays----
		rel_error[d] = (Mean - tdst)/tdst
		maps[d]  = MAP
		times[d] = int_time
		sds[d]   = SD
		cis[d,:] = CI

		#---- plot just random sample

		if d in random_sample:
			y_min,y_max= 0.95*np.min(sample),1.05*np.max(sample)

			fig = plt.figure(221, figsize=(6.3, 6.3))
			ax0 = fig.add_subplot(223, position=rect_scatter)
			ax0.set_xlabel("Iteration")
			ax0.set_ylabel("Distance [pc]")
			ax0.set_ylim(y_min,y_max)
			ax0.plot(sample.T, '-', color='k', alpha=0.3,linewidth=0.3)
			ax0.axhline(tdst,  color='black',ls="solid",linewidth=0.8,label="True")
			ax0.axhline(MAP,   color='blue',ls="--",linewidth=0.5,label="MAP")
			ax0.axhline(CI[0], color='blue',ls=":",linewidth=0.5,label="CI 95%")
			ax0.axhline(CI[1], color='blue',ls=":",linewidth=0.5)
			ax0.legend(loc="upper left",ncol=4,fontsize=4)


			ax1 = fig.add_subplot(224, position=rect_histy)
			ax1.set_ylim(y_min,y_max)
			ax1.axhline(tdst,  color='black',ls="solid",linewidth=0.8,label="True")
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
		bar.update(d) 
	pdf.close()

	#---------- return data frame with errors and fractiona uncertainties----
	data = pn.DataFrame(np.column_stack((true_dst,pax,u_pax,ru_pax,maps,rel_error,sds,cis[:,0],cis[:,1],times)),
		columns=['True distance','Parallax','Parallax,uncertainty','Fractional uncertainty',
		         'Observed distance','Fractional error','Standard deviation','p2.5%','p97.5%','Autocorrelation time'])
	return data

data = syn_validation(N_samples,data_loc=data_loc,data_scale=data_scale)
data.sort_values('Fractional uncertainty',inplace=True)

pdf = PdfPages(filename=dir_graphs+"Errors.pdf")
nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.7
bottom, height = 0.1, 0.7
bottom_h = left_h = left + width + 0.01

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)
axHistx   = plt.axes(rect_histx)
axHisty   = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

asymetric_uncert = [np.abs(data['Observed distance']-data['p2.5%'])/data['True distance'],np.abs(data['p97.5%']-data['Observed distance'])/data['True distance']]

# the scatter plot:
axScatter.errorbar(x=data["Fractional uncertainty"],y=data["Fractional error"],yerr=asymetric_uncert,
	elinewidth=0.1,alpha=0.1,fmt=".",ecolor="grey",color="blue",ms=0.1)
axScatter.scatter(x=data["Fractional uncertainty"],y=data["Fractional error"],color="blue",s=0.5)
axScatter.set_xlabel("Fractional uncertainty")
axScatter.set_ylabel("Fractional error")

axHistx.hist(data["Fractional uncertainty"], color="blue",bins=100,density=True)
axHisty.hist(data["Fractional error"], color="blue",bins=100,density=True,orientation="horizontal")

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

plt.text(1.2,0.9,s="MAD "+'{0:.3f}'.format(np.mean(np.abs(data["Fractional error"]))),fontsize=8,
	horizontalalignment='center',verticalalignment='center', transform=axHistx.transAxes)
plt.text(1.2,0.7,s="Mean "+'{0:.3f}'.format(np.mean(data["Fractional error"])),fontsize=8,
	horizontalalignment='center',verticalalignment='center', transform=axHistx.transAxes)
plt.text(1.2,0.5,s="SD "+'{0:.3f}'.format(np.std(data["Fractional error"])),fontsize=8,
	horizontalalignment='center',verticalalignment='center', transform=axHistx.transAxes)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()

pdf = PdfPages(filename=dir_graphs+"AutoTimes.pdf")
plt.hist(data["Autocorrelation time"], bins=100)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()


# ids = np.where(data["Fractional error"]< 0)
# print data.iloc[ids]



        

