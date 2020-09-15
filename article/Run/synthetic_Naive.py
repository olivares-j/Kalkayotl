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
	along with Kalkayotl.  If not, see <http://www.gnu.org/licenses/>.
'''
#------------ LOAD LIBRARIES -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines

plt.rcParams.update({'font.size': 20})
figsize = (15,15)

#============ Directories =============================
#-------Main directory ---------------
dir_main  = "/home/javier/Repositories/Kalkayotl/article/"
# dir_main  = os.getcwd() +"/"
# dir_main  = "/raid/jromero/Kalkayotl/"
#----------- Data --------------------
dir_data  = dir_main + "Synthetic/"
dir_outs  = dir_main + "Outputs/Naive/"
file_plot = dir_outs + "naive.pdf"


#------- Create directories -------
os.makedirs(dir_outs,exist_ok=True)


name = "Gaussian"

random_states = [1,2,3,4,5,6,7,8,9,10]
distances = [100 ,200 ,300 ,400 ,500 ,600 ,700 ,800 ,900 ,1000,2000,3000,4000,5000]
smp_size = 1000

quantiles = [0.025,0.975]

list_of_sources = [
			{"number":100, "color":"orange"},
			{"number":500, "color":"seagreen"},
			{"number":1000,"color":"cornflowerblue"}
			]

parameters = [
				{"name":"Location","xlim":[90,5000],"ylim":[[-0.11,0.11],[-0.01,0.16],[0,101]]},
				# {"name":"Scale",   "xlim":[90,2100],"ylim":[[-0.15,0.75],[0.01,0.26], [0.0,101]]}
				]

iocs      = [
			{"name":"arithmetic",  "linestyle" : "-"},
			{"name":"weighted","linestyle" : "--"},
			]



pars      = np.zeros((3,3,1,len(list_of_sources),len(distances),len(random_states)))

for n,sources in enumerate(list_of_sources):
	for r,random_state in enumerate(random_states):
		for d,distance in enumerate(distances):

			#------------ Local data -----------------------------------------------------
			file_data = dir_data + name + "_" + str(sources["number"]) + "_" + str(random_state) +"/" + name + "_" + str(distance) + ".csv"

			data = pd.read_csv(file_data,usecols=["parallax","parallax_error"])

			#------------ Mean --------------------------
			mu_plx = np.mean(data["parallax"])
			sd_plx = np.std(data["parallax"])
			e_mu_plx = sd_plx/np.sqrt(sources["number"])
			samples_mean = np.random.normal(loc=mu_plx,
				scale=e_mu_plx,size=smp_size)
			distances_mean = 1000./samples_mean
			mean_dst = 1000./mu_plx
			q_mean_dst = np.quantile(distances_mean,quantiles)

			pars[0,0,0,n,d,r] = (mean_dst - distance)/distance
			pars[1,0,0,n,d,r] = 0.5*(q_mean_dst[1]-q_mean_dst[0])/distance
			pars[2,0,0,n,d,r] = distance >= q_mean_dst[0] and distance < q_mean_dst[1]
			#--------------------------------------------



			#------- Average -----------------------------------
			weights = 1./(data["parallax_error"]**2)
			norm_weights = weights/np.sum(weights)
			avg_plx,sum_wgts = np.average(data["parallax"],
						weights=norm_weights,
						returned=True)
			e_avg_plx = np.sqrt(np.dot(norm_weights,data["parallax_error"]**2))
			samples_avg = np.random.normal(loc=avg_plx,
				scale=e_avg_plx,size=smp_size)
			distances_avg = 1000./samples_avg
			avg_dst = 1000./avg_plx
			q_avg_dst = np.quantile(distances_avg,quantiles)

			pars[0,1,0,n,d,r] = (avg_dst - distance)/distance
			pars[1,1,0,n,d,r] = 0.5*(q_avg_dst[1]-q_avg_dst[0])/distance
			pars[2,1,0,n,d,r] = distance >= q_avg_dst[0] and distance < q_avg_dst[1]
			#---------------------------------------------------

			

			# #--------------- Sample mean --------------------------------------
			# samples = []
			# for i,datum in data.iterrows():
			# 	samples.append(np.random.normal(loc=datum["parallax"],
			# 		scale=datum["parallax_error"],size=smp_size))

			# dsts = 1000./np.concatenate(samples)
			# mu_dst = np.mean(dsts)
			# sd_dst = np.std(dsts)
			# e_mu_dst = sd_dst/np.sqrt(sources["number"])

			# pars[0,2,0,n,d,r] = (mu_dst - distance)/distance
			# pars[1,2,0,n,d,r] = e_mu_dst/distance
			# pars[2,2,0,n,d,r] = np.abs(mu_dst - distance) <= 2.*e_mu_dst
			# #---------------------------------------------------------------


			



#============================== Plots ========================================================

line_corr = [mlines.Line2D([], [],  color="black",
								linestyle=ioc["linestyle"],
								label=ioc["name"]) for ioc in iocs]

line_numbers = [mlines.Line2D([], [],  color=sources["color"],
								linestyle="-",
								label=str(sources["number"])) for sources in list_of_sources]

pdf = PdfPages(filename=file_plot)
fig, axes = plt.subplots(num=0,nrows=3, ncols=1, sharex='col',sharey=False,figsize=figsize)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.0)

axes[0].set_ylabel('Fractional error')
axes[1].set_ylabel('Fractional uncertainty')
axes[2].set_ylabel('Credibility [%]')

for p,par in enumerate(parameters):
	axes[0].set_title(par["name"])
	axes[2].set_xlabel("Distance [pc]")
	axes[2].set_xscale('log')
	axes[2].set_xlim(par["xlim"])

	axes[0].set_ylim(par["ylim"][0])
	axes[1].set_ylim(par["ylim"][1])
	axes[2].set_ylim(par["ylim"][2])

	# axes[1,p].set_yscale("log")



	for i,ioc in enumerate(iocs):
		for n,scs in enumerate(list_of_sources): 
			mu_ferror = np.mean(pars[0,i,p,n],axis=1)
			sd_ferror = np.std( pars[0,i,p,n],axis=1)
			mu_span   = np.mean(pars[1,i,p,n],axis=1)
			sd_span   = np.std( pars[1,i,p,n],axis=1)
			mu_cred   = np.mean(pars[2,i,p,n],axis=1)*100.
			sd_cred   = np.std( pars[2,i,p,n],axis=1)*100.

			#--------------- Accuracy -------------------------------------------------------------------
			axes[0].fill_between(distances,y1=mu_ferror-sd_ferror,y2=mu_ferror+sd_ferror,
								color=scs["color"],linestyle=ioc["linestyle"],alpha=0.1,label=None)
			axes[0].plot(distances,mu_ferror,color=scs["color"],alpha=0.8,linestyle=ioc["linestyle"],label=None)

			#---------------- Precision -----------------------------------------------------------------
			axes[1].fill_between(distances,y1=mu_span-sd_span,y2=mu_span+sd_span,
								color=scs["color"],linestyle=ioc["linestyle"],alpha=0.1,label=None)
			axes[1].plot(distances,mu_span,color=scs["color"],alpha=0.8,linestyle=ioc["linestyle"],label=None)

			#---------------- Credibility -----------------------------------------------------------------
			axes[2].fill_between(distances,y1=(mu_cred-sd_cred),y2=mu_cred+sd_cred,
									color=scs["color"],linestyle=ioc["linestyle"],alpha=0.1,label=None)
			axes[2].plot(distances,mu_cred,color=scs["color"],alpha=0.8,linestyle=ioc["linestyle"],label=None)

			
axes[0].legend(
	title="Type of mean",
	handles=line_corr,
	shadow = False,
	bbox_to_anchor=(0.17,-0.02, 0.30, 0.1),
	bbox_transform = fig.transFigure,
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 2,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 10
	)
fig.legend(title="Number of sources",
	handles=line_numbers,
	shadow = False,
	bbox_to_anchor=(0.57,-0.02, 0.3, 0.1),
	bbox_transform = fig.transFigure,
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 3,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 10)

pdf.savefig(bbox_inches='tight')
plt.close(0)
pdf.close()

			