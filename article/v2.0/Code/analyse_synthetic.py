'''
Copyright 2023 Javier Olivares Romero

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
import seaborn as sns
import dill

# dill.load_session(str(sys.argv[1]))
dill.load_session("./globals_Gaussian.pkl")
list_of_n_stars = [100,200,400]
list_of_distances = [100,200,400,800,1600]
list_of_seeds = [0,1,2,3,4]
family = "Gaussian"
dimension = 6
velocity_model = "linear"
append = "" if (dimension == 3) and (velocity_model == "joint") else "_1E+06"

#---------------------- Directories and data -------------------------------
dir_main  = "/home/jolivares/Repos/Kalkayotl/article/v2.0/Synthetic/"
dir_plots = "/home/jolivares/Dropbox/MisArticulos/Kalkayotl/Figures/"

dir_data  = "{0}{1}_{2}/".format(dir_main,family,velocity_model)
base_data = dir_data  + family + "_n{0}_d{1}_s{2}.csv"
base_dir  = dir_data  + "{0}D_{1}".format(dimension,family) + "_n{0}_d{1}_s{2}_{3}"+append+"/"
base_plt  = "{0}{1}_{2}/".format(dir_plots,family,velocity_model)


file_plot_src = base_plt + "{0}D_{1}_{2}_source-level.pdf".format(dimension,family,velocity_model)
file_plot_grp = base_plt + "{0}D_{1}_{2}_group-level.pdf".format(dimension,family,velocity_model)
file_plot_cnv = base_plt + "{0}D_{1}_{2}_convergence.pdf".format(dimension,family,velocity_model)
file_plot_rho = base_plt + "{0}D_{1}_{2}_correlation.pdf".format(dimension,family,velocity_model)
file_data_all = dir_data + "{0}D_data.h5".format(dimension)

do_all_dta = True
do_plt_cnv = True
do_plt_grp = True
do_plt_src = True
do_plt_rho = True
#---------------------------------------------------------------------------

coordinates = ["X","Y","Z","U","V","W"][:dimension]
true_src_columns = sum([["source_id"],coordinates],[])
obs_src_columns = sum([["source_id"],
					["mean_"+c for c in coordinates],
					["hdi_2.5%_"+c for c in coordinates],
					["hdi_97.5%_"+c for c in coordinates],
					],[])

obs_grp_columns = ["Parameter","mean","hdi_2.5%","hdi_97.5%","ess_bulk","r_hat"]
par_grp_names = sum([
					["{0}D::loc[{1}]".format(dimension,c) for c in coordinates],
					["{0}D::std[{1}]".format(dimension,c) for c in coordinates],
					],[])
par_lin_names = sum([
					["6D::kappa[{0}]".format(i) for i in range(3)],
					["6D::omega[{0}, {1}]".format(i,j) for i in range(2) for j in range(3)]
					],[])

#-------------- True parameters --------------------------------
true_loc_grp  = np.array([0.0,0.0,0.0,10.,10.,10.])[:dimension]
true_sds_grp  = np.array([9.,9.,9.,1.,1.,1.])[:dimension]
true_grp_pars = np.hstack([true_loc_grp,true_sds_grp])[1:]
true_lin_pars = np.array([1.,1.,1.,-1.,-1.,-1.,1.,1.,1.])

if velocity_model == "linear":
	true_grp_names = sum([par_grp_names,par_lin_names],[])
	true_grp_pars = np.hstack([true_grp_pars,true_lin_pars])
else:
	true_grp_names = par_grp_names

units = {}
for name in true_grp_names:
	if (("X" in name) or ("Y" in name) or ("Z" in name)):
		units[name] = "pc"
	elif "U" in name or "V" in name or "W" in name :
		units[name] = "$\\rm{km\\,s^{-1}}$" 
	else:
		units[name] = "$\\rm{km\\,s^{-1}\\,pc^{-1}}$"
	
true_grp_names.pop(0)
#---------------------------------------------------------------

#----------------------- Gaussian ---------------------------------------------------------------
if family == "Gaussian":
	parameters = [
				{"name":"Location","xlim":[90,5000],"ylim":[[-0.11,0.11],[0.0005,0.15],[0,101]]},
				{"name":"Scale",   "xlim":[90,2100],"ylim":[[-0.15,0.75],[0.01,0.25],  [0.0,101]]}
				]
#------------------------------------------------------------------------------------------------

#----------------------- Gaussian ---------------------------------------------------------------
if family == "GMM":
	parameters = [
				{"name":"Location", "xlim":[90,5000],"ylim":[[-0.19,0.09],[0.0005,0.15],[0,101]]},
				{"name":"Scale",    "xlim":[90,5000],"ylim":[[-0.69,0.99],[0.01,0.99],  [0.0,101]]},
				{"name":"Amplitude","xlim":[90,5000],"ylim":[[-0.85,0.85],[-0.19,1.25],   [0.0,101]]}
				]
#------------------------------------------------------------------------------------------------

#------------------------Statistics -----------------------------------
sts_grp = [
		{"key":"err", "name":"Error",            "ylim":None},
		{"key":"unc", "name":"Uncertainty",      "ylim":None},
		{"key":"crd", "name":"Credibility",  "ylim":[0,100]},
		]
sts_src = [
		{"key":"rms", "name":"RMS",              "ylim":None},
		{"key":"unc", "name":"Uncertainty",      "ylim":None},
		{"key":"crd", "name":"Credibility",  "ylim":[0,100]},
		{"key":"rho", "name":"Correlation",      "ylim":None},
		]
sts_cnv = [
		{"key":"ess_bulk", "name":"ESS",         "ylim":None},
		{"key":"r_hat",    "name":"$\\hat{R}$",  "ylim":None},
		]
#-----------------------------------------------------------------------

if do_all_dta:
	#-------------- True parameters -----------------------
	df_true_grp = pn.DataFrame(data=true_grp_pars,
					columns=["true"],
					index=true_grp_names).rename_axis(
					index="Parameter")
	#-------------------------------------------------------------

	dfs_grp = []
	dfs_sts = []
	dfs_src = []
	for n,n_stars in enumerate(list_of_n_stars):
		for d,distance in enumerate(list_of_distances):
			for s,seed in enumerate(list_of_seeds):
				#------------- Parametrization --------------------
				if distance <= 500.:
					parametrization = "central"
				else:
					parametrization = "non-central"
				#-------------------------------------------

				#------------- Files ------------------------------
				dir_chains = base_dir.format(n_stars,int(distance),seed,parametrization)
				file_obs_grp   = dir_chains  + "Cluster_statistics.csv"
				file_obs_src   = dir_chains  + "Sources_statistics.csv"
				file_true_src  = base_data.format(n_stars,int(distance),seed)
				#--------------------------------------------------

				#-------------------- Read sources -------------------------------
				df_true_src = pn.read_csv(file_true_src, usecols=true_src_columns)
				df_true_src.set_index("source_id",inplace=True)

				df_obs_src = pn.read_csv(file_obs_src, usecols=obs_src_columns)
				df_obs_src.set_index("source_id",inplace=True)
				#------------------------------------------------------------------

				#---------------- Read parameters ----------------------------
				df_obs_grp = pn.read_csv(file_obs_grp,usecols=obs_grp_columns)
				df_obs_grp.set_index("Parameter",inplace=True)
				#--------------------------------------------------------------

				df_true_grp.loc["{0}D::loc[X]".format(dimension),"true"] = distance

				#---------- Join ------------------------
				df_grp = pn.merge(
								left=df_obs_grp,
								right=df_true_grp,
								left_index=True,
								right_index=True)

				df_src = pn.merge(
								left=df_obs_src,
								right=df_true_src,
								left_index=True,
								right_index=True)
				#----------------------------------------

				#---------------------------- Parameter statistics -----------------------------------------------------
				df_grp["err"] = df_grp.apply(lambda x: (x["mean"] - x["true"]),  axis = 1)
				df_grp["unc"] = df_grp.apply(lambda x: (x["hdi_97.5%"]-x["hdi_2.5%"]),  axis = 1)
				df_grp["crd"] = df_grp.apply(lambda x: 100.*((x["true"] >= x["hdi_2.5%"]) & (x["true"] <= x["hdi_97.5%"])),
											axis = 1)
				#----------------------------------------------------------------------------------------------------

				#----------------------------- Sources statistics ----------------------------------------------------------
				true_loc = df_grp.loc[["{0}D::loc[{1}]".format(dimension,coord) for coord in coordinates],"true"].to_numpy()
				df_src["r_ctr"] = df_src.apply(lambda x: np.sqrt(np.sum((
												np.array([x[coord] for coord in coordinates])-true_loc)**2)),
												axis = 1)
				
				dfs_sts_tmp = []
				dfs_src_tmp = []
				for i,coord in enumerate(coordinates):
					mean = "mean_{0}".format(coord)
					low  = "hdi_2.5%_{0}".format(coord)
					up   = "hdi_97.5%_{0}".format(coord)

					df_src[coord+"_ctr"] = df_src.apply(lambda x: (x[coord] - true_loc[i]),  axis = 1)
					df_src[coord+"_err"] = df_src.apply(lambda x: (x[mean] - x[coord]),  axis = 1)
					df_src[coord+"_unc"] = df_src.apply(lambda x: (x[up]-x[low]),  axis = 1)
					df_src[coord+"_in_"] = df_src.apply(lambda x: 100.*((x[coord] >= x[low]) & (x[coord] <= x[up])),
													  axis = 1)

					dt_src = {}
					dt_src["ctr"] = df_src[coord+"_ctr"].values
					dt_src["err"] = df_src[coord+"_err"].values
					dt_src["unc"] = df_src[coord+"_unc"].values/2.0

					tmp = pn.DataFrame(data=dt_src,index=pn.MultiIndex.from_product([df_src.index,[coord]],
	                           names=['source_id', 'Parameter'])).reset_index()
					dfs_src_tmp.append(tmp)

					st_src = {}
					st_src["rms"] = np.sqrt(np.mean(df_src[coord+"_err"]**2))
					st_src["unc"] = np.mean(df_src[up]-df_src[low])
					st_src["crd"] = np.mean(df_src[coord+"_in_"])
					st_src["rho"] = np.corrcoef(df_src[coord+"_err"],df_src[coord+"_ctr"])[0,1]

					tmp = pn.DataFrame(data=st_src,index=[coord]).rename_axis(index="Parameter").reset_index()
					dfs_sts_tmp.append(tmp)

				df_sts = pn.concat(dfs_sts_tmp,ignore_index=True)
				df_src = pn.concat(dfs_src_tmp,ignore_index=True)
				# -----------------------------------------------------------------------------------------------------

				#------------ Case -------------
				df_grp["n_stars"] = n_stars
				df_sts["n_stars"] = n_stars
				df_src["n_stars"] = n_stars

				df_grp["distance"] = distance
				df_sts["distance"] = distance
				df_src["distance"] = distance

				df_grp["seed"] = seed
				df_sts["seed"] = seed
				df_src["seed"] = seed
				#--------------------------------

				#----------- Append ----------------
				dfs_grp.append(df_grp)
				dfs_sts.append(df_sts)
				dfs_src.append(df_src)
				#------------------------------------

	#------------ Concatenate --------------------
	df_grp = pn.concat(dfs_grp,ignore_index=False)
	df_sts = pn.concat(dfs_sts,ignore_index=True)
	df_src = pn.concat(dfs_src,ignore_index=True)
	#--------------------------------------------

	#---------------- Group-level statisitcs --------------------
	dfg_grp = df_grp.groupby(["Parameter","n_stars","distance"],sort=False)
	df_grp_hdi  = pn.merge(
				left=dfg_grp.quantile(q=0.025),
				right=dfg_grp.quantile(q=0.975),
				left_index=True,
				right_index=True,
				suffixes=("_low","_up"))
	df_sts_grp  = pn.merge(
				left=dfg_grp.mean(),
				right=df_grp_hdi,
				left_index=True,
				right_index=True).reset_index()
	#------------------------------------------------------------

	#---------- Source-level statistics -------------------------
	dfg_sts = df_sts.groupby(["Parameter","n_stars","distance"],sort=False)
	df_sts_hdi  = pn.merge(
				left=dfg_sts.quantile(q=0.025),
				right=dfg_sts.quantile(q=0.975),
				left_index=True,
				right_index=True,
				suffixes=("_low","_up"))
	df_sts_src  = pn.merge(
				left=dfg_sts.mean(),
				right=df_sts_hdi,
				left_index=True,
				right_index=True).reset_index()
	#------------------------------------------------------------

	#------------ Save data --------------------------
	df_sts_grp.to_hdf(file_data_all,key="df_sts_grp")
	df_sts_src.to_hdf(file_data_all,key="df_sts_src")
	df_grp.to_hdf(file_data_all,key="df_grp")
	df_src.to_hdf(file_data_all,key="df_src")
	#-------------------------------------------------
else:
	#------------ Read data --------------------------------
	df_sts_grp = pn.read_hdf(file_data_all,key="df_sts_grp")
	df_sts_src = pn.read_hdf(file_data_all,key="df_sts_src")
	df_grp     = pn.read_hdf(file_data_all,key="df_grp")
	df_src     = pn.read_hdf(file_data_all,key="df_src")
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
						col_wrap=3,
						hue="n_stars")
		fg.map(sns.scatterplot,"distance",st["key"])
		fg.add_legend()
		fg.set_axis_labels("Distance [pc]",st["name"])
		# fg.set(xscale="log")
		pdf.savefig(bbox_inches='tight')
		plt.close()
	pdf.close()
	#-------------------------------------------------------------------------

if do_plt_grp:
	#---------------- Group-level --------------------------------------------
	
	#------------- Split into all and linear models --------------------------
	mask_lin = df_sts_grp['Parameter'].str.contains("kappa|omega",regex=True)
	df_lin = df_sts_grp.loc[mask_lin]
	df_all = df_sts_grp.loc[~mask_lin]
	#-------------------------------------------------------------------------

	pdf = PdfPages(filename=file_plot_grp)
	for st in sts_grp:
		fg = sns.FacetGrid(data=df_all,
						col="Parameter",
						sharey=False,
						margin_titles=True,
						col_wrap=3,
						hue="n_stars")
		fg.map(sns.lineplot,"distance",st["key"])
		fg.map(plt.fill_between,"distance",
				st["key"]+"_low",
				st["key"]+"_up",
				alpha=0.1)
		fg.add_legend()

		#------------ Units ----------------------------
		axs = fg.axes_dict
		for par in df_all["Parameter"]:
			if st["name"] == "Credibility":
				unit = "[%]"
			elif st["name"] == "Correlation":
				unit = ""
			else:
				unit = "[pc]" if any([x in par for x in ["X","Y","Z"]]) else "[$\\rm{km\\,s^{-1}}$]"
			axs[par].set_xlabel("Distance [pc]")
			if ("X" in par) or ("U" in par):
				axs[par].set_ylabel("{0} {1}".format(st["name"],unit))
			axs[par].title.set_text(par)
		#-----------------------------------------------

		pdf.savefig(bbox_inches='tight')
		plt.close()

		if not df_lin.empty:
			fg = sns.FacetGrid(data=df_lin,
							col="Parameter",
							sharey=False,
							margin_titles=True,
							col_wrap=3,
							hue="n_stars")
			fg.map(sns.lineplot,"distance",st["key"])
			fg.map(plt.fill_between,"distance",
					st["key"]+"_low",
					st["key"]+"_up",
					alpha=0.1)
			fg.add_legend()

			#------------ Units ----------------------------
			axs = fg.axes_dict
			for i,par in enumerate(df_lin["Parameter"]):
				if st["name"] == "Credibility":
					unit = "[%]"
				elif st["name"] == "Correlation":
					unit = ""
				else:
					unit = "[$\\rm{km\\,s^{-1}\\,pc^{-1}}$]"
				axs[par].set_xlabel("Distance [pc]")
				if i in [0,3,6]:
					axs[par].set_ylabel("{0} {1}".format(st["name"],unit))
				axs[par].title.set_text(par)
			#-----------------------------------------------

			pdf.savefig(bbox_inches='tight')
			plt.close()

	pdf.close()
	#-------------------------------------------------------------------------

if do_plt_src:
	#-------------- Source level----------------------------------------------
	pdf = PdfPages(filename=file_plot_src)
	for st in sts_src:
		fg = sns.FacetGrid(data=df_sts_src,
						col="Parameter",
						sharey=False,
						sharex=True,
						margin_titles=True,
						col_wrap=3,
						hue="n_stars")
		fg.map(sns.lineplot,"distance",st["key"])
		fg.map(plt.fill_between,"distance",
				st["key"]+"_low",
				st["key"]+"_up",
				alpha=0.1)
		fg.add_legend()

		#------------ Units ----------------------------
		axs = fg.axes_dict
		for coord in coordinates:
			if st["name"] == "Credibility":
				unit = "[%]"
			elif st["name"] == "Correlation":
				unit = ""
			else:
				unit = "[pc]" if coord in ["X","Y","Z"] else "[$\\rm{km\\,s^{-1}}$]"
			axs[coord].set_xlabel("Distance [pc]")
			if coord in ["X","U"]:
				axs[coord].set_ylabel("{0} {1}".format(st["name"],unit))
			axs[coord].title.set_text(coord)
		#-----------------------------------------------
		# plt.subplots_adjust(wspace=0.2)

		pdf.savefig(bbox_inches='tight')
		plt.close()
	pdf.close()
	#-------------------------------------------------------------------------
if do_plt_rho:
	#---------- Source-level -------------------------------
	dfg_src = df_src.groupby("distance")
	#-------------------------------------------------------

	#-------------- Correlation ----------------------------------------------
	pdf = PdfPages(filename=file_plot_rho)
	for name,df in dfg_src.__iter__():
		fg = sns.FacetGrid(data=df,
						col="Parameter",
						margin_titles=True,
						sharey=False,
						sharex=False,
						col_wrap=3,
						hue="n_stars")
		fg.map(sns.scatterplot,"ctr","err",
						s=10,alpha=0.5,
						zorder=1,
						rasterized=True)
		fg.add_legend()
		fg.map(plt.errorbar,"ctr","err","unc",
						fmt="none",ecolor="gray",
						elinewidth=0.1,
						zorder=0,
						rasterized=True)

		#------------ Units ----------------------------
		axs = fg.axes_dict
		for coord in coordinates:
			if coord in ["X","Y","Z"]:
				axs[coord].set_xlabel("Offset [pc]")
			else:
				axs[coord].set_xlabel("Offset [$\\rm{km\\,s^{-1}}$]")
			if coord == "X":
				axs[coord].set_ylabel("Error [pc]")
			if coord == "U":
				axs[coord].set_ylabel("Error [$\\rm{km\\,s^{-1}}$]")
			axs[coord].title.set_text(coord)
		#-----------------------------------------------

		pdf.savefig(bbox_inches='tight',dpi=200)
		plt.close()
	pdf.close()
	sys.exit()
	#-------------------------------------------------------------------------