#------------ LOAD LIBRARIES -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import numpy as np
import pandas as pn

import matplotlib.pyplot as plt
import seaborn as sns
import dill

family = "Gaussian"
dimension = "6D"

#---------------------- Directories and data -------------------------------
dir_main  = "/home/jolivares/Repos/Kalkayotl/article/v2.0/Rup161/"
dir_plots = "/home/jolivares/Dropbox/MisArticulos/Kalkayotl/Figures/Rup161/"
dir_run   = "/6D_Gaussian_Galactic_linear_1E+06/"
authors = ["GG+2023","Tarricq+2022","GG+2023_clean"]#,"Tarricq+2022_clean"]
file_data_all = dir_main  + "Data.h5"
file_plot_cnv = dir_plots + "Rup161_convergence.png"
file_plot_grp = dir_plots + "Rup161_group-level.png"
file_plot_lnr = dir_plots + "Rup161_linear.png"

do_all_dta = True
do_plt_cnv = False
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
		#------------------------------------------------------

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
	#-------------------------------------------------
else:
	#------------ Read data --------------------------------
	df_all = pn.read_hdf(file_data_all,key="df_all")
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