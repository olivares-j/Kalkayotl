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

prior = "Uniform"


#============ Directories and data =================
dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_out    = dir_main  + "Outputs/Synthetic/"+prior+"/"
dir_data   = dir_main  + "Data/Synthetic/"
file_plot  = dir_main  + "Outputs/Plots/Accuracy_and_precision_"+prior+".pdf"
#==================================================================

true_scale = 10.0
true_gamma = 3.0
true_rt    = 5.0
true_shift = 0.1
true_scl2  = 20.0
true_fract = 0.5

random_states = [1,2,3,4,5,6,7,8,9,10]
distances = [100 ,200 ,300 ,400 ,500 ,600 ,700 ,800 ,900 ,1000,2000,3000,4000,5000]

list_of_sources = [
			{"number":100, "color":"orange"},
			{"number":500, "color":"seagreen"},
			{"number":1000,"color":"cornflowerblue"}
			]

iocs      = [
			{"name":"off","value":"indep","linestyle":"--"},
			{"name":"on", "value":"corr", "linestyle":"-"}
			]

#---------------------- Uniform -----------------------------------------------------------------------
if prior is "Uniform":
	parameters = [
				{"name":"Location","xlim":[90,5000],"ylim":[[-0.11,0.11 ],[0.0005,0.15],[0.0,101]]},
				{"name":"Scale",   "xlim":[90,2100],"ylim":[[-0.15,0.75], [0.01,0.25],  [0.0,101]]}
				]
#--------------------------------------------------------------------------------------------------

#----------------------- Gaussian ---------------------------------------------------------------
if prior is "Gaussian":
	parameters = [
				{"name":"Location","xlim":[90,5000],"ylim":[[-0.11,0.11],[0.0005,0.15],[0,101]]},
				{"name":"Scale",   "xlim":[90,2100],"ylim":[[-0.15,0.75],[0.01,0.25],  [0.0,101]]}
				]
#------------------------------------------------------------------------------------------------

#----------------------- Gaussian ---------------------------------------------------------------
if prior is "GMM":
	parameters = [
				{"name":"Location", "xlim":[90,5000],"ylim":[[-0.19,0.09],[0.0005,0.15],[0,101]]},
				{"name":"Scale",    "xlim":[90,5000],"ylim":[[-0.69,0.99],[0.01,0.99],  [0.0,101]]},
				{"name":"Amplitude","xlim":[90,5000],"ylim":[[-0.85,0.85],[-0.19,1.25],   [0.0,101]]}
				]
#------------------------------------------------------------------------------------------------


#---------------------- EFF -----------------------------------------------------------------------
if prior is "EFF":
	parameters = [
				{"name":"Location","xlim":[90,5000],"ylim":[[-0.05,0.06 ],[0.0005,0.15],[0.0,101]]},
				{"name":"Scale",   "xlim":[90,2100],"ylim":[[-0.15,0.75], [0.01,1],     [0.0,101]]},
				{"name":"Gamma",   "xlim":[90,5000],"ylim":[[0.05,0.75],  [0.01,0.46],  [0.0,101]]}
				]
#--------------------------------------------------------------------------------------------------

#---------------------- King -----------------------------------------------------------------------
if prior is "King":
	parameters = [
				{"name":"Location",     "xlim":[90,5000],"ylim":[[-0.09,0.09],[0.0005,0.15],[0.0,101]]},
				{"name":"Scale",        "xlim":[90,2100],"ylim":[[-0.25,0.75], [0.01,0.95], [0.0,101]]},
				{"name":"Tidal radius", "xlim":[90,5000],"ylim":[[-0.55,0.9],  [0.01,1.95],  [0.0,101]]}
				]
#-------------------------------------------------------------------------------------------------- 
statistics = [
			{"name":"Credibility [%]",        "ylims":[10,105]},
			{"name":"Fractional RMS",         "ylims":[0.001,0.15]},
			{"name":"Fractional uncertainty", "ylims":[0.001,0.17]},
			{"name":"Correlation",            "ylims":[-1.01,0.01]},
			]

stats     = np.zeros((5,2,len(list_of_sources),len(distances),len(random_states)))
pars      = np.zeros((3,2,len(parameters),len(list_of_sources),len(distances),len(random_states)))

for n,sources in enumerate(list_of_sources):
	for r,random_state in enumerate(random_states):
		for d,distance in enumerate(distances):
			for i,ioc in enumerate(iocs):
				#------------- Files ----------------------------------------------------------------------------------
				dir_name      = prior + "_" + str(sources["number"]) + "_" + str(random_state) +"/" + prior +"_"+str(distance)
				name          = prior + "_" + str(sources["number"]) + "_" + str(distance) + "_" + str(random_state)
				dir_chains    = dir_out     + name +"/"+ioc["value"]+"/" 
				file_par      = dir_chains  + "Cluster_mean.csv"
				file_src      = dir_chains  + "Sources_mean.csv"
				file_true     = dir_data    +  dir_name + ".csv"
				#-----------------------------------------------------------------------------------------------------

				#-------- Read data --------------------------------
				true = pn.read_csv(file_true,usecols=["ID","r","parallax_error","parallax"])
				true.sort_values(by="ID",inplace=True)
				true.set_index("ID",inplace=True)
				if prior in ["Uniform","Gaussian"]:
					true_val = np.array([distance,true_scale])
				if prior is "GMM":
					# true_val = np.array([distance,true_scale,true_fract])
					true_val = np.array([distance*(1.+true_shift),true_scl2,1.-true_fract])
				if prior is "EFF":
					true_val = np.array([distance,true_scale,true_gamma])
				if prior is "King":
					true_val = np.array([distance,true_scale,true_rt])
				#--------------------------------------------------

				# ------ Cluster parameters --------------------------------------------------------
				df_pars  = pn.read_csv(file_par,usecols=["Parameter","mean","lower","upper"])
				df_pars.set_index("Parameter",inplace=True)

				#----- Drop extra parameters (valid for GMM)----------------
				to_drop = ["_1" in a for a in df_pars.index.values]
				df_pars.drop(df_pars.index.values[to_drop], inplace=True)
				#-----------------------------------------------------------
				df_pars.insert(loc=0,column="true",value=true_val)

				df_pars["fError"]  = df_pars.apply(lambda x: (x["mean"]  - x["true"])/x["true"],  axis = 1)
				df_pars["fUncer"]  = df_pars.apply(lambda x: (x["upper"] - x["lower"])/x["true"], axis = 1)
				df_pars["In"]      = df_pars.apply(lambda x: ((x["true"]>=x["lower"]) 
					                                   and (x["true"]<=x["upper"])), axis = 1)
				# ---------------------------------------------------------------------------------

				pars[0,i,:,n,d,r] = df_pars["fError"]
				pars[1,i,:,n,d,r] = df_pars["fUncer"]*0.5
				pars[2,i,:,n,d,r] = df_pars["In"]


				#------- Observed sources -------------------------------------------
				infered  = pn.read_csv(file_src,usecols=["ID","mean","lower","upper"])
				infered.sort_values(by="ID",inplace=True)
				infered.set_index("ID",inplace=True)
				df       = true.join(infered,on="ID",lsuffix="_true",rsuffix="_obs")
				#------------------------------------------------------------------------

				# ----------- Compute offset and uncertainty -----------------------------
				df["Bias"]   = df.apply(lambda x: x["mean"]-x["r"], axis = 1)
				df["Offset"] = df.apply(lambda x: x["r"]-true_val[0], axis = 1)
				df["Frac"]   = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
				df["Span"]   = df.apply(lambda x: x["upper"] - x["lower"], axis = 1)
				df["In"]     = df.apply(lambda x: ((x["r"]>=x["lower"]) and (x["r"]<=x["upper"])), axis = 1)


				#------------ Statistics ---------------------------------------------------------
				stats[0,i,n,d,r] = 100*np.sum(df["In"])/len(df)
				stats[1,i,n,d,r] = np.sqrt(np.mean(df["Bias"]**2))/true_val[0]
				stats[2,i,n,d,r] = np.mean(0.5*df["Span"])/true_val[0]
				stats[3,i,n,d,r] = np.corrcoef(df["Offset"],df["Bias"])[0,1]

#============================== Plots ========================================================

line_corr = [mlines.Line2D([], [],  color="black",
								linestyle=ioc["linestyle"],
								label=ioc["name"]) for ioc in iocs]

line_numbers = [mlines.Line2D([], [],  color=sources["color"],
								linestyle="-",
								label=str(sources["number"])) for sources in list_of_sources]
pdf = PdfPages(filename=file_plot)
fig, axes = plt.subplots(num=0,nrows=3, ncols=len(parameters), sharex='col',sharey=False,figsize=(12,12))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.0)

axes[0,0].set_ylabel('Fractional error')
axes[1,0].set_ylabel('Fractional uncertainty')
axes[2,0].set_ylabel('Credibility [%]')

for p,par in enumerate(parameters):
	axes[0,p].set_title(par["name"])
	axes[2,p].set_xlabel("Distance [pc]")
	axes[2,p].set_xscale('log')
	axes[2,p].set_xlim(par["xlim"])

	axes[0,p].set_ylim(par["ylim"][0])
	axes[1,p].set_ylim(par["ylim"][1])
	axes[2,p].set_ylim(par["ylim"][2])

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
			axes[0,p].fill_between(distances,y1=mu_ferror-sd_ferror,y2=mu_ferror+sd_ferror,
								color=scs["color"],linestyle=ioc["linestyle"],alpha=0.1,label=None)
			axes[0,p].plot(distances,mu_ferror,color=scs["color"],alpha=0.8,linestyle=ioc["linestyle"],label=None)

			#---------------- Precision -----------------------------------------------------------------
			axes[1,p].fill_between(distances,y1=mu_span-sd_span,y2=mu_span+sd_span,
								color=scs["color"],linestyle=ioc["linestyle"],alpha=0.1,label=None)
			axes[1,p].plot(distances,mu_span,color=scs["color"],alpha=0.8,linestyle=ioc["linestyle"],label=None)

			#---------------- Credibility -----------------------------------------------------------------
			# axes[2,p].fill_between(distances,y1=(mu_cred-sd_cred),y2=mu_cred+sd_cred,
			# 					color=scs["color"],linestyle=ioc["linestyle"],alpha=0.1,label=None)
			axes[2,p].plot(distances,mu_cred,color=scs["color"],alpha=0.8,linestyle=ioc["linestyle"],label=None)

			
axes[0,0].legend(
	title="Spatial correlations",
	handles=line_corr,
	shadow = False,
	bbox_to_anchor=(0.17,0.0, 0.25, 0.1),
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
	bbox_to_anchor=(0.57,0.0, 0.3, 0.1),
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
# pdf.close()
# sys.exit()

fig, axes = plt.subplots(num=1,nrows=len(statistics), ncols=1, sharex=True,figsize=(6,12))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.0)
for s,st in enumerate(statistics):
	for n,nsc in enumerate(list_of_sources):
		for i,ioc in enumerate(iocs):
			mu_sts = np.mean(stats[s,i,n],axis=1)
			sd_sts = np.std( stats[s,i,n],axis=1)
			axes[s].fill_between(distances,y1=mu_sts-sd_sts,y2=mu_sts+sd_sts,
				color=nsc["color"],linestyle=ioc["linestyle"],alpha=0.1,label=None)
			axes[s].plot(distances,mu_sts,color=nsc["color"],alpha=0.8,linestyle=ioc["linestyle"])
	axes[s].set_ylabel(st["name"])
	axes[s].set_ylim(st["ylims"])
plt.xlabel('Distance [pc]')
plt.xscale('log')
plt.xlim(parameters[0]["xlim"])

axes[0].legend(
	title="Spatial correlations",
	handles=line_corr,
	shadow = False,
	bbox_to_anchor=(0.1,0.85, 0.4, 0.1),
	bbox_transform = fig.transFigure,
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 2,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 10)
fig.legend(title="Number of sources",
	handles=line_numbers,
	shadow = False,
	bbox_to_anchor=(0.5,0.85, 0.4, 0.1),
	bbox_transform = fig.transFigure,
	borderaxespad=0.,
	frameon = True,
	fancybox = True,
	ncol = 3,
	fontsize = 'smaller',
	mode = 'expand',
	loc = 10)
pdf.savefig(bbox_inches='tight')
plt.close(1)

pdf.close()
