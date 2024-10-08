#------------ LOAD LIBRARIES -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import numpy as np
import pandas as pn

import matplotlib.pyplot as plt
import seaborn as sns
import dill

sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 2.5})

family = "Gaussian"
dimension = "6D"

#---------------------- Directories and data -------------------------------
dir_main  = "/home/jolivares/Repos/Kalkayotl/article/v2.0/Hyades/Oh+2020/"
dir_plots = "/home/jolivares/Dropbox/MisArticulos/Kalkayotl/Figures/Hyades/"
dir_tabs = "/home/jolivares/Dropbox/MisArticulos/Kalkayotl/Tables/"
dir_run   = "/6D_Gaussian_ICRS_linear_1E+06/"
cases = ["GDR3","GDR2"]
file_data_all = dir_main  + "Data.h5"
file_plot_cnv = dir_plots + "Hyades_convergence.png"
file_plot_grp = dir_plots + "Hyades_group-level.png"
file_plot_lnr = dir_plots + "Hyades_linear.png"
file_tab_grp  = dir_tabs  + "Hyades_group-level.tex"

do_all_dta = False
do_plts    = False
do_tab_grp = True
#---------------------------------------------------------------------------

coordinates = ["X","Y","Z","U","V","W"]
obs_grp_columns = ["Parameter","mean","sd","hdi_2.5%","hdi_97.5%","ess_bulk","r_hat"]

#------------------------Statistics -----------------------------------
sts_cnv = [
		{"key":"ess_bulk", "name":"ESS",         "ylim":None},
		{"key":"r_hat",    "name":"$\\hat{R}$",  "ylim":None},
		]

parameters_grp = [
"6D::loc[U]",
"6D::loc[V]",
"6D::loc[W]",
"6D::std[U]",
"6D::std[V]",
"6D::std[W]",
"6D::corr_vel[0, 1]",
"6D::corr_vel[0, 2]",
"6D::corr_vel[1, 2]"
]
parameters_lnr = {
"corr_vel[0, 1]":"$\\rho_{UV}$",
"corr_vel[0, 2]":"$\\rho_{UW}$",
"corr_vel[1, 2]":"$\\rho_{VW}$",
"kappa":"$||\\kappa||$",
"omega_x":"$\\omega_x$",
"omega_y":"$\\omega_y$",
"omega_z":"$\\omega_z$",
"w_1":"$w_1$",
"w_2":"$w_2$",
"w_3":"$w_3$",
"w_4":"$w_4$",
"w_5":"$w_5$"
}

def remove_6D(x):
	return x.replace("6D::","")

def format_mu_sd(mu,sd):
	if np.isnan(sd):
		return "${0:2.2f}$".format(mu)
	else:
		return "${0:2.2f}\\pm{1:2.2f}$".format(mu,sd)

mapper_lnr = {}
for par in parameters_lnr.keys():
	mapper_lnr[par+" [m.s-1.pc-1]"] = par


columns_fmt = [
			("loc[X]","[pc]"),
			("loc[Y]","[pc]"),
			("loc[Z]","[pc]"),
			("loc[U]","$\\rm{[km\\, s^{-1}]}$"),
			("loc[V]","$\\rm{[km\\, s^{-1}]}$"),
			("loc[W]","$\\rm{[km\\, s^{-1}]}$"),
			("std[X]","[pc]"),
			("std[Y]","[pc]"),
			("std[Z]","[pc]"),
			("std[U]","$\\rm{[km\\, s^{-1}]}$"),
			("std[V]","$\\rm{[km\\, s^{-1}]}$"),
			("std[W]","$\\rm{[km\\, s^{-1}]}$"),
			("$\\rho_{UV}$",""),
			("$\\rho_{UW}$",""),
			("$\\rho_{VW}$",""),
			("$||\\kappa||$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$"),
			("$\\omega_x$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$"),
			("$\\omega_y$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$"),
			("$\\omega_z$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$"),
			("$w_1$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$"),
			("$w_2$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$"),
			("$w_3$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$"),
			("$w_4$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$"),
			("$w_5$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$")
			]
#-----------------------------------------------------------------------

#------------ Literature values --------------------
data_lit ={"Oh+2020":{  "6D::loc[U]":[-6.144,0.029,-6.086,-6.036],
						"6D::loc[V]":[45.539,0.050,45.629,45.724],
						"6D::loc[W]":[5.471,0.025,5.518,5.563],
						"6D::std[U]":[0.304,0.070,0.442,0.561],
						"6D::std[V]":[0.352,0.017,0.383,0.414],
						"6D::std[W]":[0.270,0.056,0.371,0.470],
						"6D::corr_vel[0, 1]":[-0.837,0.370,-0.146,0.502],
						"6D::corr_vel[0, 2]":[-0.561,0.297,-0.015,0.536],
						"6D::corr_vel[1, 2]":[-0.455,0.171,-0.165,0.161],
						"kappa":[-18.385,6.417,-6.5,5.434],
						"omega_x":[-6.570,5.513,3.270,14.245],
						"omega_y":[-15.803,9.779,2.236,21.091],
						"omega_z":[-20.170,8.713,-4.440,12.407],
						"w_1":[-9.05,5.461,1.447,11.613],
						"w_2":[-25.598,10.074,-6.589,12.016],
						"w_3":[-15.204,8.696,1.656,17.449],
						"w_4":[-41.540,15.598,-11.191,18.310],
						"w_5":[-0.938,6.322,10.643,23.013]
						},
			}
authors = [key for key,value in data_lit.items()]

if do_all_dta:
	#================== Literature parameters -=========================
	df_lit = []
	for author in authors:
		#---------------- Group level ----------------------------
		tmp = pn.DataFrame.from_dict(data=data_lit[author],
									orient="index",
									columns=["3%","sd","mean","97%"])
		tmp.set_index(pn.MultiIndex.from_product(
					[[author],tmp.index.values]),
					# names=["Author","Parameter"],
					inplace=True)

		#------------ Lower and upper limits --------------------------------
		tmp["low"] = tmp.apply(lambda x:x["mean"]-x["3%"],axis=1)
		tmp["up"]  = tmp.apply(lambda x:x["97%"]-x["mean"],axis=1)
		#--------------------------------------------------------------------

		df_lit.append(tmp.reset_index())
		#--------------------------------------------------------

	df_lit = pn.concat(df_lit,ignore_index=True)
	df_lit.set_index(["level_0","level_1"],inplace=True)
	df_lit.index.set_names(["Author","Parameter"],inplace=True)
	#==================================================================

	#================== Inferred parameters ===================================

	#---------------------- Loop over cases -------------------------------
	dfs_grp = []
	for case in cases:
		#------------- Files ----------------------------------
		dir_chains = dir_main + case + dir_run
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
		df_grp.set_index("Parameter",inplace=True)
		
		df_grp.set_index(pn.MultiIndex.from_product(
			[[case],df_grp.index.values]),
			# names=["Author","Parameter"],
			inplace=True)
		#-------------------------------------------------------------

		dfs_grp.append(df_grp)
	df_grp = pn.concat(dfs_grp,ignore_index=False)
	#---------------------------------------------------------------------

	#------------ Lower and upper limits ----------------------------------
	df_grp["low"] = df_grp.apply(lambda x:x["mean"]-x["hdi_2.5%"],axis=1)
	df_grp["up"]  = df_grp.apply(lambda x:x["hdi_97.5%"]-x["mean"],axis=1)
	#----------------------------------------------------------------------
	#=================================================================================

	#----------- Set join on and reset indices ------------------
	df_grp.index.set_names(["Author","Parameter"],inplace=True)
	df_all = pn.concat([df_grp,df_lit],ignore_index=False)
	df_all.reset_index(level=0,inplace=True)
	#-----------------------------------------------------------

	#------------ Save data --------------------------
	df_all.to_hdf(file_data_all,key="df_all")
	#-------------------------------------------------
else:
	#------------ Read data --------------------------------
	df_all = pn.read_hdf(file_data_all,key="df_all")
	#-------------------------------------------------------

#=========================== Plots =======================================

if do_plts:
	for file_plt,parameters in zip([file_plot_lnr,file_plot_grp],[parameters_lnr.keys(),parameters_grp]):

		tmp_all = df_all.loc[parameters]
		tmp_all.reset_index(inplace=True)

		#---------------- Group-level --------------------------------------------
		fg = sns.FacetGrid(data=tmp_all,
						col="Parameter",
						sharey=True,
						sharex=False,
						margin_titles=True,
						col_wrap=3,
						hue="Author",
						height=2,
						aspect=1.5,
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

if do_tab_grp:

	df_all.rename(columns={"Author":"Origin"},inplace=True)
	tab = df_all.loc[:,["Origin","mean","sd"]]

	tab.replace(0.0,value=np.nan,inplace=True)

	tab["string"] = tab.apply(lambda x: format_mu_sd(x["mean"],x["sd"]),axis=1)
	tab.drop(columns=["mean","sd"],inplace=True)
	tab = tab.pivot(columns="Origin",values="string")
	tab.rename(axis=0,mapper=remove_6D,inplace=True)
	tab.rename(index=parameters_lnr,inplace=True)

	tab = tab.loc[[col[0] for col in columns_fmt],["Oh+2020","GDR2","GDR3"]]
	
	tab.index = pn.MultiIndex.from_tuples(columns_fmt)
	tab.index.names = ["Parameter","Units"]
	tab.columns.name = None

	#---------- Table to latex ---------------
	s = tab.style
	s.format(precision=1,na_rep="-",
		escape="latex-math")
	print(s.to_latex(
		column_format="llrrr",
		multicol_align="c",
		hrules=True,
		clines=None))
	# sys.exit()
	#--------------- Save -----------------------------------------------
	s.to_latex(file_tab_grp,
		column_format="llrrr",
		multicol_align="c",
		hrules=True,
		clines=None)
	#---------------------------------------------------------------------
