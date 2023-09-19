#------------ LOAD LIBRARIES -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import numpy as np
import pandas as pn

import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import arviz as az
import dill

family = "Gaussian"
dimension = "6D"

#---------------------- Directories and data -------------------------------
dir_main  = "/home/jolivares/Repos/Kalkayotl/article/v2.0/Alessi13/"
dir_plots = "/home/jolivares/Dropbox/MisArticulos/Kalkayotl/Figures/Alessi13/"
dir_run   = "/6D_Gaussian_Galactic_linear_1E+06/"
authors = ["GG+2023","GG+2023_core","Galli+2021","Galli+2021_core"]
file_data_all = dir_main  + "Data.h5"
file_plot_cnv = dir_plots + "Alessi13_convergence.png"
file_plot_grp = dir_plots + "Alessi13_group-level.png"
file_plot_lnr = dir_plots + "Alessi13_linear.png"

do_all_dta = True
do_plt_cnv = False
do_plt_age = True
#---------------------------------------------------------------------------

coordinates = ["X","Y","Z","U","V","W"]
obs_grp_columns = ["Parameter","mean","sd","hdi_2.5%","hdi_97.5%","ess_bulk","r_hat"]

#------------------------Statistics -----------------------------------
sts_cnv = [
		{"key":"ess_bulk", "name":"ESS",         "ylim":None},
		{"key":"r_hat",    "name":"$\\hat{R}$",  "ylim":None},
		]

parameters_grp = [
"6D::loc[X]",
"6D::loc[Y]",
"6D::loc[Z]",
"6D::loc[U]",
"6D::loc[V]",
"6D::loc[W]",
"6D::std[X]",
"6D::std[Y]",
"6D::std[Z]",
"6D::std[U]",
"6D::std[V]",
"6D::std[W]"
]
parameters_lnr = [
"kappa",
"omega_x",
"omega_y",
"omega_z",
"w_1",
"w_2",
"w_3",
"w_4",
"w_5"
]
mapper_lnr = {}
for par in parameters_lnr:
	mapper_lnr[par+" [m.s-1.pc-1]"] = par
#-----------------------------------------------------------------------

if do_all_dta:
	#================== Inferred parameters ===================================

	#---------------------- Loop over cases -------------------------------
	dfs_grp = []
	for author in authors:
		#------------- Files ----------------------------------
		dir_chains = dir_main + author + dir_run
		file_jnt   = dir_chains  + "Cluster_statistics.csv"
		file_lnr   = dir_chains  + "Lindegren_velocity_statistics.csv"
		file_smp   = dir_chains  + "Samples.h5"
		#------------------------------------------------------

		#------------- Read posterior samples of Kappa -------------
		with h5py.File(file_smp,'r') as hf:
			kappa = np.array(hf.get("Cluster/6D::kappa"))
		df_kappa = pn.DataFrame(data=kappa,columns=["X","Y","Z"])
		#-----------------------------------------------------------

		#---------------- Read parameters ----------------------------
		df_jnt = pn.read_csv(file_jnt,usecols=obs_grp_columns)
		df_lnr = pn.read_csv(file_lnr)
		df_lnr.set_index("Parameter",inplace=True)
		df_lnr.rename(index=mapper_lnr,inplace=True)
		df_lnr.reset_index(inplace=True)
		df_grp = pn.concat([df_jnt,df_lnr],ignore_index=True)
		df_grp["Author"] = author
		#-------------------------------------------------------------

		dfs_grp.append(df_grp)

	df_all = pn.concat(dfs_grp,ignore_index=True)
	#---------------------------------------------------------------------

	#------------ Lower and upper limits ----------------------------------
	df_all["low"] = df_all.apply(lambda x:x["mean"]-x["hdi_2.5%"],axis=1)
	df_all["up"]  = df_all.apply(lambda x:x["hdi_97.5%"]-x["mean"],axis=1)
	#----------------------------------------------------------------------
	#=================================================================================

	df_all.set_index("Parameter",inplace=True)

	#------------ Save data --------------------------
	df_all.to_hdf(file_data_all,key="df_all")
	df_kappa.to_hdf(file_data_all,key="df_kappa")
	#-------------------------------------------------
else:
	#------------ Read data --------------------------------
	df_all = pn.read_hdf(file_data_all,key="df_all")
	df_kappa = pn.read_hdf(file_data_all,key="df_kappa")
	#-------------------------------------------------------

#=========================== Plots =======================================
for file_plt,parameters in zip([file_plot_lnr,file_plot_grp],[parameters_lnr,parameters_grp]):

	tmp_all = df_all.loc[parameters]
	tmp_all.reset_index(inplace=True)

	#---------------- Group-level --------------------------------------------
	fg = sns.FacetGrid(data=tmp_all,
					col="Parameter",
					sharey=True,
					sharex=False,
					margin_titles=True,
					col_wrap=3,
					hue="Author"
					)
	fg.map(sns.scatterplot,"mean","Author",s=50)
	fg.set_axis_labels("Value","")
	axs = fg.axes_dict
	dfg_all = tmp_all.groupby("Parameter")
	for parameter in parameters:
		ax = axs[parameter]
		dfa = dfg_all.get_group(parameter)

		#----------- Literature HDI values -----------------------
		ax.errorbar(x=dfa["mean"],y=dfa["Author"],
					xerr=dfa.loc[:,["low","up"]].to_numpy().T,
					capsize=0,elinewidth=1,capthick=0,
					fmt="none",
					color="tab:grey",
					zorder=0)
		#----------------------------------------------------

		#------ Literature sd values -------------------------------
		ax.errorbar(x=dfa["mean"],y=dfa["Author"],
					xerr=dfa["sd"],
					capsize=5,elinewidth=1,capthick=1,
					fmt="none",
					ms=15,
					barsabove=True,
					ecolor="tab:gray",
					zorder=0)
		#------------------------------------------------------


	plt.savefig(file_plt,bbox_inches='tight')
	plt.close()
	#-------------------------------------------------------------------------

if do_plt_age:
	def age(kappa):
		return  1./(1.022712165*kappa)

	
	df_kappa["mu"] = df_kappa.apply(lambda row:np.mean([row["X"],row["Y"]]),axis=1)
	df_age = pn.DataFrame(data={"Age":age(df_kappa["mu"])})
	print("Age from the simple mean of Kx and Ky: {0:2.1f} +- {1:2.1f}".format(
		df_age.median().to_numpy()[0],df_age.std().to_numpy()[0]))
	#---------------------------------------------------------------------------------
	
	#----------- Transform to age -----------------------------------------------
	df_age = pn.concat([
		pn.DataFrame(data={"Age":age(kappa[:,0]),"Coordinate":"X"}),
		pn.DataFrame(data={"Age":age(kappa[:,1]),"Coordinate":"Y"})],
		ignore_index=True)
	#----------------------------------------------------------------------------

	dfg = df_age.groupby("Coordinate")
	smp_x = dfg.get_group("X").drop(columns="Coordinate").to_numpy().flatten()
	smp_y = dfg.get_group("Y").drop(columns="Coordinate").to_numpy().flatten()

	mu_x = np.median(smp_x)
	mu_y = np.median(smp_y)

	mu_kx = np.median(kappa[:,0])
	mu_ky = np.median(kappa[:,1])

	limits_x = np.zeros((2,2))
	limits_y = np.zeros((2,2))
	wmus     = np.zeros(2)
	wsds     = np.zeros((2,2))
	for i,hdi_prob in enumerate([0.68,0.95]):
		print("------------ HDI prob = {0} ---------------------".format(hdi_prob))
		hdi_x = az.hdi(smp_x,hdi_prob=hdi_prob)
		hdi_y = az.hdi(smp_y,hdi_prob=hdi_prob)
		
		limits_x[i] = hdi_x - mu_x
		limits_y[i] = hdi_y - mu_y

		sd_kx = np.mean(np.abs(az.hdi(kappa[:,0],hdi_prob=hdi_prob)-mu_kx))
		sd_ky = np.mean(np.abs(az.hdi(kappa[:,1],hdi_prob=hdi_prob)-mu_ky))

		print("Kx: {0:2.1f}+-{1:2.1f}".format(mu_kx*1000.,sd_kx*1000.))
		print("Ky: {0:2.1f}+-{1:2.1f}".format(mu_ky*1000.,sd_ky*1000.))

		means = np.array([mu_kx,mu_ky])
		variances = np.array([sd_kx**2,sd_ky**2])
		weights  = 1./variances
		
		weighted_variance = 1/np.sum(weights)
		
		mu = weighted_variance*np.sum(means*weights)
		sd = np.sqrt(weighted_variance)
		tau = age(mu)
		hdi_tau = age(np.array([mu+sd,mu-sd]))

		print("Weighted average of Kappa: {0:2.1f}+-{1:2.1f}".format(mu*1000.,sd*1000.))

		wsds[i] = hdi_tau-tau
		wmus[i] = tau
		print("------------------------------------------------")
	print()
	print("Age from inverted Kx (1sigma): {0:2.1f} {1:2.1f}+{2:2.1f}".format(mu_x,limits_x[0,0],limits_x[0,1]))
	print("Age from inverted Kx (2sigma): {0:2.1f} {1:2.1f}+{2:2.1f}".format(mu_x,limits_x[1,0],limits_x[1,1]))
	print()
	print("Age from inverted Ky (1sigma): {0:2.1f} {1:2.1f}+{2:2.1f}".format(mu_y,limits_y[0,0],limits_y[0,1]))
	print("Age from inverted Ky (2sigma): {0:2.1f} {1:2.1f}+{2:2.1f}".format(mu_y,limits_y[1,0],limits_y[1,1]))
	print()
	print("Age from weighted mean of Kx and Ky (1sigma): {0:2.1f} {1:2.1f}+{2:2.1f}".format(wmus[0],wsds[0,0],wsds[0,1]))
	print("Age from weighted mean of Kx and Ky (2sigma): {0:2.1f} {1:2.1f}+{2:2.1f}".format(wmus[1],wsds[1,0],wsds[1,1]))