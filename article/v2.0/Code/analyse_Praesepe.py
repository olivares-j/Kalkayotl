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
dir_main  = "/home/jolivares/Projects/Kalkayotl/Praesepe/"
dir_plots = "/home/jolivares/Dropbox/MisArticulos/Kalkayotl/Figures/"
dir_tabs  = "/home/jolivares/Dropbox/MisArticulos/Kalkayotl/Tables/"
dir_run   = "/6D_Gaussian_Galactic_linear_1E+06/"
authors = ["Hao+2022","GG+2023","Jadhav+2024","Hao+2022_wtr","GG+2023_wtr","Jadhav+2024_wtr"]#,"GG+2023_core","Hao+Lodieu","GG+Lodieu"]
file_data_all = dir_main  + "Data.h5"
file_plot_cnv = dir_plots + "Praesepe_convergence.png"
file_plot_grp = dir_plots + "Praesepe_group-level.png"
file_plot_lnr = dir_plots + "Praesepe_linear.png"
file_tab_grp  = dir_tabs  + "Praesepe_group-level.tex"

do_all_dta = True
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
parameters_lnr = {
"kappa":"$||\\kappa||$",
"kappa_x":"$\\kappa_x$",
"kappa_y":"$\\kappa_y$",
"kappa_z":"$\\kappa_z$",
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
			("$||\\kappa||$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$"),
			("$\\kappa_x$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$"),
			("$\\kappa_y$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$"),
			("$\\kappa_z$", "$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$"),
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
		df_grp["Author"] = author.split("_")[0]
		df_grp["Case"] = "Cleaned \& <$R_{tidal}$" if "wtr" in author else "Original"
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
	tab = df_all.loc[:,["Origin","Case","mean","sd"]]

	tab.replace(0.0,value=np.nan,inplace=True)

	tab["string"] = tab.apply(lambda x: format_mu_sd(x["mean"],x["sd"]),axis=1)
	tab.drop(columns=["mean","sd"],inplace=True)
	tab.reset_index(inplace=True)
	tab.set_index(["Parameter","Case","Origin"],inplace=True)
	# tab = tab.pivot(columns=["Case","Origin"],values="string")

	print(tab)
	tab = tab.unstack(level=["Case","Origin"])
	tab = tab.droplevel(0, axis=1) 
	tab.rename(axis=0,mapper=remove_6D,inplace=True)
	tab.rename(index=parameters_lnr,inplace=True)
	# tab.rename(columns={"GG+2023_wrt":"Clean GG+2023",
	# 			'Hao+2022_wtr':'Hao+2022 WTR'},
	# 			inplace=True)

	tab = tab.loc[[col[0] for col in columns_fmt],:]
	
	tab.index = pn.MultiIndex.from_tuples(columns_fmt)
	tab.index.names = ["Parameter","Units"]
	tab.columns.name = None

	#---------- Table to latex ---------------
	s = tab.style
	s.format(precision=1,na_rep="-",
		escape="latex-math")
	print(s.to_latex(
		column_format="llrrrrrr",
		multicol_align="c",
		hrules=True,
		clines=None))
	# sys.exit()
	#--------------- Save -----------------------------------------------
	s.to_latex(file_tab_grp,
		column_format="llrrrrrr",
		multicol_align="c",
		hrules=True,
		clines=None)
	#---------------------------------------------------------------------
