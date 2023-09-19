#------------ LOAD LIBRARIES -------------------
from __future__ import absolute_import, unicode_literals, print_function
import os
import sys
import numpy as np
import pandas as pn
import pymc as pm
import arviz as az

import matplotlib.pyplot as plt
import seaborn as sns
import dill

#---------------------- Directories and data -------------------------------
dir_main  = "/home/jolivares/Repos/Kalkayotl/article/v2.0/"
dir_plots = "/home/jolivares/Dropbox/MisArticulos/Kalkayotl/Figures/"
dir_run   = dir_main  + "{0}/{1}/6D_Gaussian_Galactic_linear_1E+06/"
file_data = dir_main  + "All_data.h5"
file_fit  = dir_main  + "All_fit.h5"
file_plot = dir_plots + "All_linear.png"
file_ylim = dir_plots + "All_ylims.pkl"
file_plft = dir_plots + "Fit_{0}_hdi{1}.png"
#---------------------------------------------------------------------------
do_read = True
do_fit  = True
save_ylim = False

groups = [
{"name":"BetaPic", "age":20,  "author":"Miret-Roig+2020"},
{"name":"IC4665",  "age":31,  "author":"Miret-Roig+2019_core"},
{"name":"IC4665",  "age":31,  "author":"GG+2023_core"},
{"name":"Alessi13","age":40,  "author":"Galli+2021_core"},
{"name":"Alessi13","age":40,  "author":"GG+2023_core"},
{"name":"Stock2",  "age":400, "author":"GG+2023_rvs_core"},
{"name":"Stock2",  "age":400, "author":"Tarricq+2022_rvs_core"},
{"name":"Hyades",  "age":640, "author":"Oh+2020/GDR3"},
{"name":"Praesepe","age":670, "author":"Hao+2022_core"},
{"name":"Praesepe","age":670, "author":"GG+2023_core"},
{"name":"Praesepe","age":670, "author":"Hao+Lodieu"},
{"name":"Praesepe","age":670, "author":"GG+Lodieu"},
{"name":"Rup147",  "age":2500,"author":"Olivares+2019_core"},
{"name":"Rup147",  "age":2500,"author":"GG+2023_core"},
]

parameters = [
"kappa",
"omega",
"Txx",
"omega_x",
"Tyy",
"omega_y",
"Tzz",
"omega_z"
]
mapper_lnr = {}
for par in parameters:
	if par == "omega":
		mapper_lnr["Rot [m.s-1.pc-1]"] = par
	mapper_lnr[par+" [m.s-1.pc-1]"] = par
#-----------------------------------------------------------------------


#================== Inferred parameters ===================================
if do_read:
	#---------------------- Loop over cases -------------------------------
	dfs_grp = []
	for group in groups:
		#------------- Files -------------------------------------------------
		dir_chains = dir_run.format(group["name"],group["author"])
		file_lnd   = dir_chains  + "Lindegren_velocity_statistics.csv"
		file_lnr   = dir_chains  + "Linear_velocity_statistics.csv"
		#---------------------------------------------------------------------

		#---------------- Read parameters ----------------------------
		df_grp = pn.concat([pn.read_csv(file_lnr),pn.read_csv(file_lnd)],
							ignore_index=True)
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
	df_all.to_hdf(file_data,key="df_all")

else:
	df_all = pn.read_hdf(file_data,key="df_all")

if do_fit:
	print("-------------- Fitting ------------------------------")

	dfg_all = df_all.groupby("Parameter")
	for parameter in parameters:
		print("......................... {0} ..................................".format(parameter))
		dfa = dfg_all.get_group(parameter)
		basic_model = pm.Model()

		with basic_model:

			# Prior independiente para cada parámetro.
			a = pm.Normal("a", mu=0, sigma=10)
			b = pm.Normal("b", mu=0, sigma=100)

			# Valor esperado
			y = a * np.log10(dfa["Age"]) + b

			# Verosimilitud de las observaciones
			y_obs = pm.Normal("y_obs", mu=y, sigma=dfa["sd"], observed=dfa["mean"])

			# draw 1000 posterior samples
			posterior = pm.sample(progressbar=False)
			
			#------------------- Print and plot posterior---------------------
			print(az.summary(posterior, round_to=3))
			plt.figure()
			for prob in [0.9545,0.9973,0.99994]:
				az.plot_posterior(posterior,hdi_prob=prob,round_to=4)
				plt.savefig(file_plft.format(parameter,prob),
					dpi=200,bbox_inches="tight")
				plt.close()
			#--------------------------------------------------------

			#------------- Save ----------------------------------------
			df_tmp = posterior.to_dataframe(groups="posterior")
			df_tmp.to_hdf(file_fit,key=parameter)
			#-----------------------------------------------------------
			print(".......................................................")

#=========================== Plots =======================================
#---------------- ylim ------------------
if save_ylim:
	ylim = {}
else:
	with open(file_ylim, 'rb') as file:
	    ylim = dill.load(file)
#----------------------------------------

#---------------- Group-level --------------------------------------------
fg = sns.FacetGrid(data=df_all,
				col="Parameter",
				sharey=False,
				sharex=True,
				margin_titles=False,
				col_wrap=2,
				hue="Group",
				)
fg.map(sns.scatterplot,"Age","mean",s=50)
fg.set_axis_labels("Age [Myr]","$\\rm{[m\\,s^{-1}\\,pc^{-1}]}$")
fg.set(xscale="log")#,ylim=(-75,100))
fg.add_legend()
axs = fg.axes_dict
dfg_all = df_all.groupby("Parameter")
x = np.linspace(10,2500,num=100)
for parameter in parameters:
	ax = axs[parameter]
	dfa = dfg_all.get_group(parameter)
	df_fit = pn.read_hdf(file_fit,key=parameter)
	df_mean = df_fit.mean()
	smp = df_fit.sample(n=100)

	#----------- Fit ------------------------------------------
	for a,b in zip(smp["a"],smp["b"]):
		ax.plot(x,a*np.log10(x) + b, color="orange", alpha=0.05,zorder=0)
	ax.plot(x,df_mean["a"]*np.log10(x) + df_mean["b"], color="red", alpha=0.5,zorder=0)
	#----------------------------------------------------------

	#----------- Literature HDI values -----------------------
	ax.errorbar(x=dfa["Age"],y=dfa["mean"],
				yerr=dfa.loc[:,["low","up"]].to_numpy().T,
				capsize=0,
				elinewidth=1,capthick=0,
				fmt="none",
				color="black",
				zorder=0)
	#----------------------------------------------------

	#------ Literature sd values -------------------------------
	ax.errorbar(x=dfa["Age"],y=dfa["mean"],
				yerr=dfa["sd"],
				capsize=5,
				elinewidth=0.5,
				capthick=0.5,
				fmt="none",
				ms=15,
				barsabove=True,
				ecolor="black",
				zorder=0)
	#------------------------------------------------------

	#------------Limits ------------------
	if save_ylim:
		ylim[parameter] = ax.get_ylim()
	else:
		ax.set_ylim(ylim[parameter])
	#-------------------------------------


plt.savefig(file_plot,bbox_inches='tight',dpi=300)
plt.close()
#-------------------------------------------------------------------------

if save_ylim:
	with open(file_ylim, 'wb') as file:
	    dill.dump(ylim, file)