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
dir_main  = "/home/jolivares/Repos/Kalkayotl/article/v2.0/BetaPic/"
dir_plots = "/home/jolivares/Dropbox/MisArticulos/Kalkayotl/Figures/BetaPic/"
dir_run   = "/6D_Gaussian_Galactic_joint_1E+06/"
authors = ["Crundall_2019","Miret-Roig_2020","Couture_2023"]
file_data_all = dir_main  + "Data.h5"
file_lite_src = dir_main  + "Couture_2023/core_sources.csv"
file_plot_src = dir_plots + "BetaPic_source-level.png"
file_plot_grp = dir_plots + "BetaPic_group-level.png"
file_plot_cnv = dir_plots + "BetaPic_convergence.png"

do_all_dta = True
do_plt_cnv = False
do_plt_grp = False
do_plt_src = True
#---------------------------------------------------------------------------

coordinates = ["X","Y","Z","U","V","W"]
obs_src_columns = sum([["source_id"],
					["mean_"+c for c in coordinates],
					["sd_"+c for c in coordinates],
					# ["hdi_2.5%_"+c for c in coordinates],
					# ["hdi_97.5%_"+c for c in coordinates],
					],[])

obs_grp_columns = ["Parameter","mean","sd","hdi_2.5%","hdi_97.5%","ess_bulk","r_hat"]
true_grp_names = sum([
					["{0}::loc[{1}]".format(dimension,c) for c in coordinates],
					["{0}::std[{1}]".format(dimension,c) for c in coordinates],
					],[])


#------------------------Statistics -----------------------------------
sts_grp = [
		{"key":"err", "name":"Error [pc]",        "ylim":None},
		{"key":"unc", "name":"Uncertainty [pc]",  "ylim":None},
		{"key":"crd", "name":"Credibility [%]",  "ylim":[0,100]},
		]
sts_src = [
		{"key":"rms", "name":"RMS [pc]",          "ylim":None},
		{"key":"unc", "name":"Uncertainty [pc]",  "ylim":None},
		{"key":"crd", "name":"Credibility [%]",  "ylim":[0,100]},
		{"key":"rho", "name":"Correlation",      "ylim":None},
		]
sts_cnv = [
		{"key":"ess_bulk", "name":"ESS",         "ylim":None},
		{"key":"r_hat",    "name":"$\\hat{R}$",  "ylim":None},
		]

parameters = sum([
	["6D::loc[{0}]".format(x) for x in coordinates],
	["6D::std[{0}]".format(x) for x in coordinates]
	], [])
#-----------------------------------------------------------------------

#------------ Literature values --------------------
data_lit ={"Crundall_2019":{"6D::loc[X]":[3.1,30.0,3.2],
							"6D::loc[Y]":[2.8,-5.5,2.8],
							"6D::loc[Z]":[1.7,7.5,1.7],
							"6D::loc[U]":[0.2,1.5,0.2],
							"6D::loc[V]":[0.1,-3.5,0.1],
							"6D::loc[W]":[0.1,-1.6,0.1],
							"6D::std[X]":[1.4,24.5,1.5],
							"6D::std[Y]":[1.1,21.6,1.2],
							"6D::std[Z]":[0.8,13.7,0.9],
							"6D::std[U]":[0.1,1.2,0.1],
							"6D::std[V]":[0.1,0.9,0.1],
							"6D::std[W]":[0.1,1.0,0.1]
							},
		   "Miret-Roig_2020":{
		   					"6D::loc[X]":[0.11,47.49,0.11],
		   					"6D::loc[Y]":[0.04,-7.89,0.04],
		   					"6D::loc[Z]":[0.05,-17.92,0.05],
							"6D::loc[U]":[0.24,-8.74,0.24],
							"6D::loc[V]":[0.11,-16.16,0.11],
							"6D::loc[W]":[0.11,-9.98,0.11],
							"6D::std[X]":[0.0,16.04,0.0],
							"6D::std[Y]":[0.0,13.18,0.0],
							"6D::std[Z]":[0.0,7.44,0.0],
							"6D::std[U]":[0.0,1.49,0.0],
							"6D::std[V]":[0.0,0.54,0.0],
							"6D::std[W]":[0.0,0.70,0.0]
							},
			"Couture_2023":{
							"6D::loc[X]":[0.0,22.691,0.0],
							"6D::loc[Y]":[0.0,-4.308,0.0],
							"6D::loc[Z]":[0.0,-18.492,0.0],
							"6D::loc[U]":[0.0,-10.2,0.0],
							"6D::loc[V]":[0.0,-15.7,0.0],
							"6D::loc[W]":[0.0,-8.64,0.0],
							"6D::std[X]":[0.0,29.698,0.0],
							"6D::std[Y]":[0.0,13.94,0.0],
							"6D::std[Z]":[0.0,8.106,0.0],
							"6D::std[U]":[0.0,1.5,0.0],
							"6D::std[V]":[0.0,0.6,0.0],
							"6D::std[W]":[0.0,0.76,0.0]
							},

			}

LSR = np.array([0., 0.,25.,11.1, 12.24,7.25])
if do_all_dta:
	#================== Literature parameters -=========================
	df_lit = []
	for author in authors:
		#---------------- Group level ----------------------------
		tmp = pn.DataFrame.from_dict(data=data_lit[author],
									orient="index",
									columns=["low","mean","up"])
		if author == "Crundall_2019":
			tmp.loc[parameters[:6],"mean"] -= LSR
		tmp.set_index(pn.MultiIndex.from_product(
					[[author],tmp.index.values]),
					# names=["Author","Parameter"],
					inplace=True)
		df_lit.append(tmp.reset_index())
		#--------------------------------------------------------

	df_lit = pn.concat(df_lit,ignore_index=True)

	#----------- source level ---------------------
	df_lit_src = pn.read_csv(file_lite_src)
	df_lit_src.set_index("source_id",inplace=True)
	#----------------------------------------------

	#--------------- Join into coordinates --------------------------
	dfs_src = []
	for coord in coordinates:
		tmp = df_lit_src.loc[:,["mean_"+coord,"sd_"+coord]]
		tmp.rename(columns={"mean_"+coord:"mean","sd_"+coord:"sd"},
					inplace=True)
		tmp.set_index(pn.MultiIndex.from_product(
					[[coord],tmp.index.values]),
					# names=["Author","Parameter"],
					inplace=True)
		dfs_src.append(tmp)
	df_lit_src = pn.concat(dfs_src,ignore_index=False)
	df_lit_src.index.set_names(["Coordinate","source_id"],inplace=True)
	#------------------------------------------------------------------
	#==================================================================

	#================== Inferred parameters ===================================
	dfs_grp = []
	for author in authors:
		#------------- Files ----------------------------------
		dir_chains = dir_main + author + dir_run
		file_grp   = dir_chains  + "Cluster_statistics.csv"
		file_src   = dir_chains  + "Sources_statistics.csv"
		#------------------------------------------------------

		if author == "Couture_2023":
			#-------------------- Read sources -------------------------------
			df_inf_src = pn.read_csv(file_src, usecols=obs_src_columns)
			df_inf_src.set_index("source_id",inplace=True)
			#------------------------------------------------------------------

			#--------------- Join into coordinates --------------------------
			dfs_src = []
			for coord in coordinates:
				tmp = df_inf_src.loc[:,["mean_"+coord,"sd_"+coord]]
				tmp.rename(columns={"mean_"+coord:"mean","sd_"+coord:"sd"},
							inplace=True)
				tmp.set_index(pn.MultiIndex.from_product(
							[[coord],tmp.index.values]),
							# names=["Author","Parameter"],
							inplace=True)
				dfs_src.append(tmp)
			df_inf_src = pn.concat(dfs_src,ignore_index=False)
			df_inf_src.index.set_names(["Coordinate","source_id"],inplace=True)
			#------------------------------------------------------------------

		#---------------- Read parameters ----------------------------
		df_grp = pn.read_csv(file_grp,usecols=obs_grp_columns)
		df_grp.set_index("Parameter",inplace=True)
		df_grp.set_index(pn.MultiIndex.from_product(
			[[author],df_grp.index.values]),
			# names=["Author","Parameter"],
			inplace=True)
		#-------------------------------------------------------------

		#------------ Lower and upper limits --------------------------------
		df_grp["low"] = df_grp.apply(lambda x:x["mean"]-x["hdi_2.5%"],axis=1)
		df_grp["up"]  = df_grp.apply(lambda x:x["hdi_97.5%"]-x["mean"],axis=1)
		#--------------------------------------------------------------------

		#----------- Append ----------------
		dfs_grp.append(df_grp.reset_index())
		#------------------------------------

	#------------ Concatenate --------------------
	df_grp = pn.concat(dfs_grp,ignore_index=True)
	#--------------------------------------------
	#=================================================================================

	#----------- Set indices -----------------------------
	df_grp.set_index(["level_0","level_1"],inplace=True)
	df_lit.set_index(["level_0","level_1"],inplace=True)
	df_grp.index.set_names(["Author","Parameter"],inplace=True)
	df_lit.index.set_names(["Author","Parameter"],inplace=True)
	#----------------------------------------------------------

	#----------------- Merge -------------
	df_src = df_inf_src.merge(df_lit_src,
		left_index=True,right_index=True,
		suffixes=("_inf","_lit"))
	#-------------------------------------

	#------------ Save data --------------------------
	df_grp.to_hdf(file_data_all,key="df_grp")
	df_src.to_hdf(file_data_all,key="df_src")
	df_lit.to_hdf(file_data_all,key="df_lit")
	#-------------------------------------------------
else:
	#------------ Read data --------------------------------
	df_grp = pn.read_hdf(file_data_all,key="df_grp")
	df_src = pn.read_hdf(file_data_all,key="df_src")
	df_lit = pn.read_hdf(file_data_all,key="df_lit")
	#-------------------------------------------------------

#=========================== Plots =======================================
if do_plt_cnv:
	#-------------- Convergence ----------------------------------------------
	pdf = PdfPages(filename=file_plot_cnv)
	for st in sts_cnv:
		fg = sns.FacetGrid(data=df_grp.reset_index(),
						col="Parameter",
						sharey=False,
						margin_titles=True,
						col_wrap=3,)
		fg.map(sns.scatterplot,"Author",st["key"])
		fg.add_legend()
		fg.set_axis_labels("Author",st["name"])
		# fg.set(xscale="log")
		pdf.savefig(bbox_inches='tight')
		plt.close()
	pdf.close()
	#-------------------------------------------------------------------------

if do_plt_grp:
	df_lit.reset_index(inplace=True)
	df_grp.reset_index(inplace=True)

	#---------------- Group-level --------------------------------------------
	fg = sns.FacetGrid(data=df_lit,
					col="Parameter",
					sharey=True,
					sharex=False,
					margin_titles=True,
					col_wrap=3,
					)
	fg.map(sns.scatterplot,"mean","Author",
			zorder=1,color="black")
	fg.set_axis_labels("Value","")
	axs = fg.axes_dict
	dfg_lit = df_lit.groupby("Parameter")
	dfg_grp = df_grp.groupby("Parameter")
	for parameter in parameters:
		ax = axs[parameter]

		#----------- Literature values -----------------------
		dfl = dfg_lit.get_group(parameter)
		ax.errorbar(x=dfl["mean"],y=dfl["Author"],
					xerr=dfl.loc[:,["low","up"]].to_numpy().T,
					capsize=5,elinewidth=3,capthick=3,
					fmt="none",color="tab:orange",zorder=0)
		#----------------------------------------------------

		#------ Inferred HDI values -------------------------------
		dfg = dfg_grp.get_group(parameter)
		ax.errorbar(x=dfg["mean"],y=dfg["Author"],
					xerr=dfg.loc[:,["low","up"]].to_numpy().T,
					capsize=0,elinewidth=1,
					fmt="none",
					ecolor="tab:green",zorder=0)
		#------------------------------------------------------

		#------ Inferred sd values -------------------------------
		dfg = dfg_grp.get_group(parameter)
		ax.errorbar(x=dfg["mean"],y=dfg["Author"],
					xerr=dfg["sd"],
					capsize=5,elinewidth=1,capthick=1,
					fmt=".",ms=15,
					barsabove=True,
					color="tab:green",
					ecolor="tab:green",zorder=0)
		#------------------------------------------------------

	plt.savefig(file_plot_grp,bbox_inches='tight')
	plt.close()
	#-------------------------------------------------------------------------

if do_plt_src:

	def annotate_rms(data, **kws):
		rms = np.sqrt(np.mean((data["mean_inf"]-data["mean_lit"])**2))
		ax = plt.gca()
		ax.text(.6, .1,"RMS : {0:2.2f}".format(rms), 
			transform=ax.transAxes,
			fontweight="normal",color="gray")

	df_src.reset_index(level="Coordinate",inplace=True)
	#-------------- Source level----------------------------------------------
	fg = sns.FacetGrid(data=df_src,
					col="Coordinate",
					sharey=False,
					sharex=False,
					margin_titles=True,
					col_wrap=3)
	fg.map(sns.scatterplot,"mean_lit","mean_inf")
	fg.set_axis_labels("Literature","This work")

	axs = fg.axes_dict
	dfg_src = df_src.groupby("Coordinate")
	for coord in coordinates:
		df = dfg_src.get_group(coord)
		ax = axs[coord]
		title = "Coordinate = {0} [{1}]".format(coord,"pc" if coord in ["X","Y","Z"] else "km/s")
		ax.title.set_text(title)

		#---------- Comparison line ---------------------------
		vmin = np.min(np.array([ax.get_xlim(),ax.get_ylim()]))
		vmax = np.max(np.array([ax.get_xlim(),ax.get_ylim()]))
		x = np.linspace(vmin,vmax,num=10)
		ax.plot(x, x, c='grey',zorder=0,linewidth=0.5)
		#------------------------------------------------------

		#----------- Literature values -----------------------
		ax.errorbar(x=df["mean_lit"],y=df["mean_inf"],
					xerr=df["sd_lit"],yerr=df["sd_inf"],
					capsize=0,elinewidth=1,capthick=0,
					fmt="none",color="gray",zorder=0)
		#----------------------------------------------------

		#--------- Annotate RMS --------
		fg.map_dataframe(annotate_rms)
		#-------------------------------

	plt.savefig(file_plot_src,bbox_inches='tight')
	plt.close()
	#-------------------------------------------------------------------------
