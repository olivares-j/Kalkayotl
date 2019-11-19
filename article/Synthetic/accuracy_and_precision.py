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

#============ Directories and data =================
dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_out    = dir_main  + "Outputs/Synthetic/"
dir_data   = dir_main  + "Data/Synthetic/"
file_plot  = dir_main  + "Outputs/Plots/Accuracy_and_precision.pdf"
#==================================================================

prior = "Gaussian"
random_states = [1,2,3,4,5]
clusters = [
			{"name":"Gaussian",     "distance":100  },
			{"name":"Gaussian",     "distance":200  },
			{"name":"Gaussian",     "distance":300  },
			{"name":"Gaussian",     "distance":400  },
			{"name":"Gaussian",     "distance":500  },
			{"name":"Gaussian",     "distance":600  },
			{"name":"Gaussian",     "distance":700  },
			{"name":"Gaussian",     "distance":800  },
			{"name":"Gaussian",     "distance":900  },
			{"name":"Gaussian",     "distance":1000 },
			{"name":"Gaussian",     "distance":1600 },
			{"name":"Gaussian",     "distance":1800 },
			{"name":"Gaussian",     "distance":2300 },
			{"name":"Gaussian",     "distance":2500 },
			{"name":"Gaussian",     "distance":3000 },
			{"name":"Gaussian",     "distance":3500 },
			{"name":"Gaussian",     "distance":4000 },
			{"name":"Gaussian",     "distance":5800 },
			]

iocs      = [
			{"name":"off","value":"indep","marker":"d","linestyle":"--"},
			{"name":"on", "value":"corr", "marker":"o","linestyle":"-"}
			]
parameters =[
			{"name":"loc","color":"brown"},
			{"name":"scl","color":"green"},

]

statistics = [
			{"name":"Credibility [%]",   "color":"green","scale":"linear"},
			{"name":"RMS [pc]",          "color":"brown","scale":"log"},
			{"name":"Source CI [pc]",    "color":"blue", "scale":"log"},
			{"name":"Parameter CI [pc]", "color":"cyan", "scale":"log"},
			{"name":"Correlation",       "color":"red",  "scale":"linear"},
			]

n_states  = len(random_states)
n_cluster = len(clusters)
distances = np.array([cluster["distance"] for cluster in clusters],dtype="int")
sources   = np.zeros((5,2,n_cluster,n_states))
pars      = np.zeros((3,2,2,n_cluster,n_states))

for r,random_state in enumerate(random_states):
	for j,cluster in enumerate(clusters):
		for i,ioc in enumerate(iocs):
			#------------- Files ----------------------------------------------------------------------------------
			dir_name      = prior + "_" + str(random_state) +"/" + cluster["name"] +"_"+str(cluster["distance"])
			dir_chains    = dir_out     + dir_name +"/"+prior+"/"+ioc["value"]+"/" 
			file_par      = dir_chains  + "Cluster_mean.csv"
			file_src      = dir_chains  + "Sources_mean.csv"
			file_true     = dir_data    +  dir_name + ".csv"
			#-----------------------------------------------------------------------------------------------------

			#-------- Read data --------------------------------
			true = pn.read_csv(file_true,usecols=["ID","r","parallax_error","parallax"])
			true.sort_values(by="ID",inplace=True)
			true.set_index("ID",inplace=True)
			true_val = np.array([np.mean(true["r"]),np.std(true["r"])])
			#--------------------------------------------------

			# ------ Cluster parameters --------------------------------------------------------
			df_pars  = pn.read_csv(file_par,usecols=["mean","lower","upper"])
			df_pars.insert(loc=0,column="true",value=true_val)

			df_pars["Bias"]  = df_pars.apply(lambda x: x["mean"]  - x["true"],  axis = 1)
			df_pars["Span"]  = df_pars.apply(lambda x: x["upper"] - x["lower"], axis = 1)
			df_pars["In"]    = df_pars.apply(lambda x: ((x["true"]>=x["lower"]) 
				                                   and (x["true"]<=x["upper"])), axis = 1)
			# ---------------------------------------------------------------------------------

			# #------- Observed sources -------------------------------------------
			# infered  = pn.read_csv(file_src,usecols=["ID","mean","lower","upper"])
			# infered.sort_values(by="ID",inplace=True)
			# infered.set_index("ID",inplace=True)
			# df       = true.join(infered,on="ID",lsuffix="_true",rsuffix="_obs")
			# #------------------------------------------------------------------------

			# #----------- Compute offset and uncertainty -----------------------------
			# df["Bias"]   = df.apply(lambda x: x["mean"]-x["r"], axis = 1)
			# df["Offset"] = df.apply(lambda x: x["r"]-true_val[0], axis = 1)
			# df["Frac"]   = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
			# df["Span"]   = df.apply(lambda x: x["upper"] - x["lower"], axis = 1)
			# df["In"]     = df.apply(lambda x: ((x["r"]>=x["lower"]) and (x["r"]<=x["upper"])), axis = 1)


			#------------ Statistics ---------------------------------------------------------
			pars[0,:,i,j,r] = df_pars["Bias"]
			pars[1,:,i,j,r] = df_pars["Span"] 
			pars[2,:,i,j,r] = df_pars["In"]

			# sources[0,i,j,r] = np.sum(df["In"])/len(df)
			# sources[1,i,j,r] = np.sqrt(np.mean(df["Bias"]**2))
			# sources[2,i,j,r] = np.mean(df["Span"])
			# sources[3,i,j,r] = np.corrcoef(df["Offset"],df["Bias"])[0,1]

#============================== Plots ========================================================

line_corr = [mlines.Line2D([], [],  color="black",
								linestyle=ioc["linestyle"],
								label=ioc["name"]) for ioc in iocs]

line_pars = [mlines.Line2D([], [],  color=par["color"],
								linestyle="-",
								label=par["name"]) for par in parameters]
pdf = PdfPages(filename=file_plot)
fig, axes = plt.subplots(num=0,nrows=2, ncols=2, sharex=True,sharey='row',figsize=(12,12))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.0)
for p,par in enumerate(parameters):
	for i,ioc in enumerate(iocs):

		#--------------- Accuracy -------------------------------------------------------------------
		mu_ferror = np.mean(pars[0,p,i],axis=1)/distances
		mi_ferror = np.min(pars[0,p,i],axis=1)/distances
		ma_ferror = np.max(pars[0,p,i],axis=1)/distances
		axes[0,p].fill_between(distances,y1=mi_ferror,y2=ma_ferror,
							color=par["color"],linestyle=ioc["linestyle"],alpha=0.3,label=None)
		axes[0,p].plot(distances,mu_ferror,color=par["color"],alpha=0.8,linestyle=ioc["linestyle"],label=None)

		#---------------- Precision -----------------------------------------------------------------
		mu_span = 0.5*np.mean(pars[1,p,i],axis=1)/distances
		mi_span = 0.5*np.min(pars[1,p,i],axis=1)/distances
		ma_span = 0.5*np.max(pars[1,p,i],axis=1)/distances
		axes[1,p].fill_between(distances,y1=mi_span,y2=ma_span,
							color=par["color"],linestyle=ioc["linestyle"],alpha=0.3,label=None)
		axes[1,p].plot(distances,mu_span,color=par["color"],alpha=0.8,linestyle=ioc["linestyle"],label=None)

		axes[p,i].set_xscale('log')
	axes[0,p].set_ylim(-0.09,0.3)
	axes[1,p].set_ylim(0.0,0.09)
	axes[1,p].set_xlabel("Distance [pc]")
	

axes[0,0].set_ylabel('Fractional error')
axes[1,0].set_ylabel('Fractional uncertainty')


		# #---------------- Credibility -----------------------------------------------------------------
		# mu_cred = np.mean(pars[2,p,i],axis=1)
		# mi_cred = np.min(pars[2,p,i],axis=1)
		# ma_cred = np.max(pars[2,p,i],axis=1)
		# # axes[2].fill_between(distances,y1=mi_cred,y2=ma_cred,
		# # 					color=par["color"],alpha=0.3,label=None)
		# axes[2].plot(distances,mu_cred,color=par["color"],alpha=0.8,linestyle=ioc["linestyle"],label=None)
	
axes[0,0].legend(
	title="Spatial correlations",
	handles=line_corr,
	shadow = False,
	bbox_to_anchor=(0.17,0.82, 0.3, 0.1),
	bbox_transform = fig.transFigure,
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 2,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper right'
	)
fig.legend(title="Parameters",
	handles=line_pars,
	shadow = False,
	bbox_to_anchor=(0.57,0.82, 0.3, 0.1),
	bbox_transform = fig.transFigure,
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 2,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper right')

pdf.savefig(bbox_inches='tight')
plt.close(0)
pdf.close()
sys.exit()




fig, axes = plt.subplots(num=1,nrows=len(statistics), ncols=1, sharex=True,figsize=(6,12))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.0)
for i,stats in enumerate(statistics):
	for j,ioc in enumerate(iocs):
		axes[i].plot(distances,sts[:,i,j],color=ioc["color"],linestyle=ioc["linestyle"])
	axes[i].set_ylabel(stats["name"])
	
	axes[i].set_yscale(stats["scale"])
plt.xlabel('Distance [pc]')
fig.legend(title="Spatial correlations",
	handles=line_corr,
	shadow = False,
	bbox_to_anchor=(0.12,0.82, 0.75, 0.1),
	bbox_transform = fig.transFigure,
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 2,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 'upper right')
pdf.savefig(bbox_inches='tight')
plt.close(1)

pdf.close()
