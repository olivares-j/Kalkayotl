import sys
import os
import numpy as np
import scipy.stats as st
import random

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec

import seaborn as sns 
import pandas as pn

from astroML.decorators import pickle_results

from p2d import p2d

dir_  = os.path.expanduser('~') +"/parallax2distance/"
dir_out = dir_ + "Analysis/"

#---------------- Reads the data --------------------
random_state = 1234
mu,sigma  = 100,10
N_samples = 100
N_iter    = 2000
prior     = str(sys.argv[1]) #"EDBJ2015", "Gaussian", "Uniform" o "Cauchy"
sg_scale  = int(sys.argv[2]) # Scale of the prior variance

#------ creates directories --------
dir_graphs = dir_out+prior+"/"+str(sg_scale)+"/"
if not os.path.isdir(dir_out+prior):
	os.mkdir(dir_out+prior)
if not os.path.isdir(dir_graphs):
	os.mkdir(dir_graphs)
#-----------------------------------

data_file = dir_graphs+"data.pkl"


@pickle_results(data_file)
def syn_validation(prior,N_samples,N_iter,mu,sigma,sg_scale):
	#---------- careate synthetic data --------
	true_dst = st.norm.rvs(loc=mu, scale=sigma, size=N_samples,random_state=random_state)
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
	
	for d,(plx,u_plx,tdst) in enumerate(zip(pax,u_pax,true_dst)):
		#------- run the p2d function ----------------------------
		MAP,Mean,SD,CI,int_time,sample = p2d(plx,u_plx,N_iter=N_iter,prior=prior,mu_prior=mu,sigma_prior=sigma,sg_prior_scale=sg_scale)
		rel_error[d] = (Mean - tdst)/tdst

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

			ax1.hist(sample.flatten(),bins=100,normed=True, 
				color="k",orientation='horizontal', fc='none', histtype='step',lw=0.5)
			pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
			plt.close()
	pdf.close()

	#---------- return data frame with errors and fractiona uncertainties----
	data = pn.DataFrame(np.column_stack((ru_pax,rel_error)),columns=['Fractional uncertainty','Fractional error'])
	return data

data = syn_validation(prior,N_samples,N_iter,mu,sigma,sg_scale)

sns.set(style="ticks", color_codes=True)
pdf = PdfPages(filename=dir_graphs+"Errors.pdf")
g = sns.JointGrid(x="Fractional error",y="Fractional uncertainty",data=data)
g = g.plot_joint(plt.scatter,s=1)
c,b,_ = g.ax_marg_x.hist(data["Fractional error"], color="b", alpha=.6,bins=100,normed=True)
_ = g.ax_marg_x.text(b[100],0.9*np.max(c),s="MAD "+'{0:.3f}'.format(np.mean(np.abs(data["Fractional error"]))),fontsize=8)
_ = g.ax_marg_x.text(b[100],0.6*np.max(c),s="Mode "+'{0:.3f}'.format(b[np.argmax(c)]+0.5*(b[2]-b[1])),fontsize=8)
_ = g.ax_marg_x.text(b[100],0.3*np.max(c),s="Mean "+'{0:.3f}'.format(np.mean(data["Fractional error"])),fontsize=8)
_ = g.ax_marg_x.text(b[100],0.1*np.max(c),s="SD "+'{0:.3f}'.format(np.std(data["Fractional error"])),fontsize=8)
_ = g.ax_marg_y.hist(data["Fractional uncertainty"], color="b", alpha=.6,orientation="horizontal",bins=100,normed=True)
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()



        

