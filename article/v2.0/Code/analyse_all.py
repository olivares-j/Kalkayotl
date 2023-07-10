#------------ LOAD LIBRARIES -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import numpy as np
import pandas as pn

import matplotlib.pyplot as plt
import seaborn as sns
import dill

#---------------------- Directories and data -------------------------------
dir_main  = "/home/jolivares/Repos/Kalkayotl/article/v2.0/{0}/{1}/"
dir_plots = "/home/jolivares/Dropbox/MisArticulos/Kalkayotl/Figures/"
dir_run   = "/6D_Gaussian_Galactic_linear_1E+06/"
file_plot = dir_plots + "All_linear.png"
#---------------------------------------------------------------------------

groups = [
{"name":"BetaPic", "age":20,  "author":"Miret-Roig_2020"},
{"name":"Alessi13","age":40,  "author":"Galli+2021_core"},
# {"name":"Alessi13","age":40,  "author":"GG+2023_core"},
{"name":"Hyades",  "age":640, "author":"Oh_2020/GDR3"},
{"name":"Praesepe","age":670, "author":"Hao+2022_core"},
{"name":"Praesepe","age":670, "author":"GG+2023_core"},
{"name":"Praesepe","age":670, "author":"Hao+Lodieu"},
{"name":"Praesepe","age":670, "author":"GG+Lodieu"},
{"name":"Rup147",  "age":2500,"author":"Olivares+2019_core"},
{"name":"Rup147",  "age":2500,"author":"GG+2023_core"},
]

parameters = [
"kappa",
"omega_x",
"omega_y",
"omega_z"
]
mapper_lnr = {}
for par in parameters:
	mapper_lnr[par+" [m.s-1.pc-1]"] = par
#-----------------------------------------------------------------------


#================== Inferred parameters ===================================

#---------------------- Loop over cases -------------------------------
dfs_grp = []
for group in groups:
	#------------- Files -------------------------------------------------
	dir_chains = dir_main.format(group["name"],group["author"]) + dir_run
	file_lnr   = dir_chains  + "Lindegren_velocity_statistics.csv"
	#---------------------------------------------------------------------

	#---------------- Read parameters ----------------------------
	df_grp = pn.read_csv(file_lnr)
	print(df_grp.columns)
	df_grp.set_index("Parameter",inplace=True)
	df_grp.rename(index=mapper_lnr,inplace=True)
	df_grp.reset_index(inplace=True)
	df_grp["Group"] = group["name"]
	df_grp["Author"] = group["author"]
	df_grp["Age"] = group["age"]
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
df_all = df_all.loc[parameters]
df_all.reset_index(inplace=True)

#=========================== Plots =======================================
#---------------- Group-level --------------------------------------------
fg = sns.FacetGrid(data=df_all,
				row="Parameter",
				sharey=False,
				sharex=True,
				margin_titles=False,
				hue="Group",
				)
fg.map(sns.scatterplot,"Age","mean",s=50)
fg.set_axis_labels("Age [Myr]","$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$")
fg.set(xscale="log")#,ylim=(-75,100))
fg.add_legend()
axs = fg.axes_dict
dfg_all = df_all.groupby("Parameter")
for parameter in parameters:
	ax = axs[parameter]
	dfa = dfg_all.get_group(parameter)

	#----------- Literature HDI values -----------------------
	ax.errorbar(x=dfa["Age"],y=dfa["mean"],
				yerr=dfa.loc[:,["low","up"]].to_numpy().T,
				capsize=0,elinewidth=1,capthick=0,
				fmt="none",
				color="tab:grey",
				zorder=0)
	#----------------------------------------------------

	#------ Literature sd values -------------------------------
	ax.errorbar(x=dfa["Age"],y=dfa["mean"],
				yerr=dfa["sd"],
				capsize=5,elinewidth=1,capthick=1,
				fmt="none",
				ms=15,
				barsabove=True,
				ecolor="tab:gray",
				zorder=0)
	#------------------------------------------------------


plt.savefig(file_plot,bbox_inches='tight',dpi=300)
plt.close()
#-------------------------------------------------------------------------