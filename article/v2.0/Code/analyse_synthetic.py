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
from matplotlib.ticker import FormatStrFormatter 
import seaborn as sns
import h5py
import arviz as az
import dill
from groups import *

pn.set_option('display.max_rows', None) 

cases = ["Gaussian_joint","Gaussian_linear","StudentT_joint","StudentT_linear","GMM_joint"]#["GMM_joint"]#["Gaussian_linear"]#

do_process = False
do_plt_grp_cmn = False
do_plt_grp_lnr = True
do_plt_grp_spc = False
do_plt_src  = False
do_plt_time = False # Only valid for StudentT_linear

file_data_all    = dir_syn + "data_cases.h5"
file_plt_grp_cmn = dir_fig + "Group-level_common.pdf"
file_plt_grp_lnr = dir_fig + "Group-level_linear.pdf"
file_plt_grp_spc = dir_fig + "Group-level_specific.pdf"
file_plt_src     = dir_fig + "Source-level.pdf"
#---------------------------------------------------------------------------

coordinates = ["X","Y","Z","U","V","W"]
true_src_columns = sum([["source_id"],coordinates],[])
obs_src_columns = sum([["source_id"],
					["mean_"+c for c in coordinates],
					["sd_"+c for c in coordinates],
					["hdi_2.5%_"+c for c in coordinates],
					["hdi_97.5%_"+c for c in coordinates],
					["label"],
					],[])

obs_grp_columns = ["Parameter","mean","sd","hdi_2.5%","hdi_97.5%","r_hat"]
#-----------------------------------------------------------------------------

#------------------------Statistics -----------------------------------
sts_grp = [
		{"key":"err", "name":"Error [%]"      , "lim_loc_pos":[-1.4,1.4],"lim_loc_vel":[-5,5] ,"lim_std_pos":[-30,30],"lim_std_vel":[-40,40]},
		{"key":"unc", "name":"Uncertainty [%]", "lim_loc_pos":[0,1.6]   ,"lim_loc_vel":[0,5]  ,"lim_std_pos":[0,40]  ,"lim_std_vel":[0,40]},
		{"key":"crd", "name":"Credibility [%]", "lim_loc_pos":[0,100]   ,"lim_loc_vel":[0,100],"lim_std_pos":[0,100] ,"lim_std_vel":[0,100]},
		]
sts_src = [
		{"key":"err", "name":"Error [%]"      ,"lim_pos":[-0.3,0.3],"lim_vel":[-5,5] },
		{"key":"unc", "name":"Uncertainty [%]","lim_pos":[0,0.5]   ,"lim_vel":[0,10]  },
		{"key":"crd", "name":"Credibility [%]","lim_pos":[0,100]   ,"lim_vel":[0,100]},
		{"key":"rho", "name":"Correlation"    ,"lim_pos":[-1,1]    ,"lim_vel":[-1,1]},
		]
#-----------------------------------------------------------------------

#-----------------------------------------------------
base_name = "n{0}_d{1}_s{2}"
dfs_grp = []
dfs_src = []
for case in cases:
	print(40*"+" +" " + case + " " +40*"+")
	if case == "Gaussian_linear":
		dir_data  = dir_syn + case + "_100/"
		assert true_signal == 0.1
	else:
		dir_data  = dir_syn + case + "/"
	file_data = dir_data + "data.h5"

	if os.path.exists(file_data) and not do_process:
		df_grp = pn.read_hdf(file_data,key="df_grp")
		df_src = pn.read_hdf(file_data,key="df_src")
	else:
		dfs_grp = []
		dfs_src = []
		for distance in list_of_distances:
			if case == "GMM_joint" and distance > 800:	
				continue
			case_args = CASE_ARGS["{0}_d{1}".format(case,int(distance))]
			for n_stars in list_of_n_stars:
				if case != "Gaussian_linear" and n_stars > 100:	
					continue
				for seed in list_of_seeds:
					tmp = "n{0}_d{1}_s{2}/".format(int(n_stars),int(distance),seed)
					print(40*"-" +" " + tmp + " " +40*"-")

					#------------- Files ----------------------------------------------------
					file_obs_grp   = dir_data   + tmp + "Cluster_statistics.csv"
					file_obs_src   = dir_data   + tmp + "Sources_statistics.csv"
					file_true_src  = dir_data   + tmp + "synthetic.csv"
					#------------------------------------------------------------------------

					#-------------- True values -----------------------
					df_true_grp = pn.DataFrame.from_dict(
										data=case_args["true_parameters"],
										orient="index",
										columns=["true"])
					df_true_grp.index.name = "Parameter"

					df_true_src = pn.read_csv(file_true_src, usecols=true_src_columns)
					df_true_src.set_index("source_id",inplace=True)
					#--------------------------------------------------------------------------

					#-------------------- Observed values -------------------------------
					df_obs_src = pn.read_csv(file_obs_src, usecols=obs_src_columns)
					df_obs_src.set_index("source_id",inplace=True)

					df_obs_grp = pn.read_csv(file_obs_grp,usecols=obs_grp_columns)
					df_obs_grp.set_index("Parameter",inplace=True)
					#------------------------------------------------------------------

					#---------------- Changes to GMM model ----------------------------------------------------
					if "GMM" in case:
						if df_obs_grp.loc["6D::weights[A]","mean"] < 0.5:
							df_obs_grp.index = df_obs_grp.index.str.replace("B","C")
							df_obs_grp.index = df_obs_grp.index.str.replace("A","B")
							df_obs_grp.index = df_obs_grp.index.str.replace("C","A")

							df_obs_src.loc[:,"label"] = df_obs_src.apply(lambda x:x["label"].replace("B","C"),axis=1)
							df_obs_src.loc[:,"label"] = df_obs_src.apply(lambda x:x["label"].replace("A","B"),axis=1)
							df_obs_src.loc[:,"label"] = df_obs_src.apply(lambda x:x["label"].replace("C","A"),axis=1)

						df_true_grp.drop(index=df_true_grp[df_true_grp.index.str.contains("B,")].index,inplace=True)
						df_true_grp.rename(index=gmm_mapper,inplace=True)

						df_obs_grp.drop(index=df_obs_grp[df_obs_grp.index.str.contains("B,")].index,inplace=True)
						df_obs_grp.rename(index=gmm_mapper,inplace=True)

						df_obs_src.drop(index=df_obs_src[df_obs_src["label"] =="B"].index,inplace=True)
					#------------------------------------------------------------------------------------------------

					#---------- Join ------------------------
					df_grp = pn.merge(
									left=df_true_grp,
									right=df_obs_grp,
									left_index=True,
									right_index=True)
					df_src = pn.merge(
									left=df_obs_src,
									right=df_true_src,
									left_index=True,
									right_index=True)
					#----------------------------------------

					#------------ Remov 6D --------------------------------------------
					df_grp.rename(index=lambda x: x.replace("6D::",""),inplace=True)
					#-----------------------------------------------------------------

					#-------------- Asses convergence ------------------
					if any(df_grp["r_hat"]>1.05):
						par = df_grp.loc[df_grp["r_hat"]>1.05]
						print("WARNING: Convergence issues at:")
						print(par)
						if any(df_grp["r_hat"]>1.1):
							par = df_grp.loc[df_grp["r_hat"]>1.1]
							print("Error: Convergence issues at:")
							print(par)
							print(300*"<")
							# sys.exit()
					#----------------------------------------------------

					#------------- Rearrengment of sources --------------------------------
					dfs_tmp = []
					for i,coord in enumerate(coordinates):
						true  = "{0}".format(coord)
						mean  = "mean_{0}".format(coord)
						lower = "hdi_2.5%_{0}".format(coord)
						upper = "hdi_97.5%_{0}".format(coord)
						std   = "sd_{0}".format(coord)

						tmp = df_src.loc[:,[true,mean,std,lower,upper]].copy()
						tmp.rename(columns={true:"true",mean:"mean",std:"sd",
										lower:"hdi_2.5%",upper:"hdi_97.5%"},
										inplace=True)

						tmp["Parameter"] = coord

						tmp.reset_index(inplace=True)
						tmp.set_index(["Parameter","source_id"],inplace=True)

						dfs_tmp.append(tmp)

					df_src = pn.concat(dfs_tmp,ignore_index=False)
					# ---------------------------------------------------------------------

					#---------------------------- Error, Uncertainty and Credibility ------------------------
					for tmp in [df_src,df_grp]:
						tmp["err"] = tmp.apply(lambda x: 100.*(x["mean"] - x["true"])/x["true"],  axis = 1)
						tmp["unc"] = tmp.apply(lambda x: 100.*(x["sd"]/np.abs(x["true"])),  axis = 1)
						tmp["crd"] = tmp.apply(lambda x: 100.*((x["true"] >= x["hdi_2.5%"]) & 
																	 (x["true"] <= x["hdi_97.5%"])),
																	axis = 1)
					#------------------------------------------------------------------------------------------

					#---------- Sources statistics ---------------------------------------------------------------------------------------------------
					for coord in coordinates:
						df_src.loc[(coord,slice(None)),"diff"] = df_src.loc[(coord,slice(None)),"true"] \
														  - df_grp.loc["loc[{0}]".format(coord),"true"]
					rho = df_src.groupby("Parameter",sort=False).apply(lambda x:np.corrcoef(x["err"],x["diff"])[0,1])

					df_src = df_src.groupby("Parameter",sort=False).median()
					df_src["rho"] = rho
					#---------------------------------------------------------------------------------------------------------------------------------

					#-------------- Select variables -----------------
					df_grp = df_grp.loc[:,["err","unc","crd"]]
					df_src = df_src.loc[:,["err","unc","crd","rho"]]
					#-------------------------------------------------

					#------------ Case -------------
					df_grp["Case"] = case
					df_src["Case"] = case

					df_grp["n_stars"] = n_stars
					df_src["n_stars"] = n_stars

					df_grp["distance"] = distance
					df_src["distance"] = distance

					df_grp["seed"] = seed
					df_src["seed"] = seed
					#--------------------------------

					df_grp.reset_index(inplace=True)
					df_src.reset_index(inplace=True)

					df_grp.set_index(["Case","Parameter","distance","n_stars","seed"],inplace=True)
					df_src.set_index(["Case","Parameter","distance","n_stars","seed"],inplace=True)

					#----------- Append ----------------
					dfs_grp.append(df_grp)
					dfs_src.append(df_src)
					#------------------------------------

		#------------ Concatenate --------------------
		df_grp = pn.concat(dfs_grp,ignore_index=False)
		df_src = pn.concat(dfs_src,ignore_index=False)
		#---------------------------------------------

		#---------------- Group-level statisitcs --------------------------------------
		dfg_grp = df_grp.groupby(["Case","Parameter","n_stars","distance"],sort=False)

		df_grp_sts  = pn.merge(
					left=dfg_grp.quantile(q=0.50),
					right=dfg_grp.std(),
					left_index=True,
					right_index=True,
					suffixes=("_mu","_sd"))
		
		df_grp_hdi  = pn.merge(
					left=dfg_grp.quantile(q=0.159),
					right=dfg_grp.quantile(q=0.841),
					left_index=True,
					right_index=True,
					suffixes=("_low","_up"))
		df_grp  = pn.merge(
					left=df_grp_sts,
					right=df_grp_hdi,
					left_index=True,
					right_index=True)
		#------------------------------------------------------------

		#---------- Source-level statistics -------------------------
		dfg_src = df_src.groupby(["Case","Parameter","n_stars","distance"],sort=False)

		df_src_sts  = pn.merge(
					left=dfg_src.quantile(q=0.50),
					right=dfg_src.std(),
					left_index=True,
					right_index=True,
					suffixes=("_mu","_sd"))

		df_src_hdi  = pn.merge(
					left=dfg_src.quantile(q=0.159),
					right=dfg_src.quantile(q=0.841),
					left_index=True,
					right_index=True,
					suffixes=("_low","_up"))

		df_src  = pn.merge(
					left=df_src_sts,
					right=df_src_hdi,
					left_index=True,
					right_index=True)
		#------------------------------------------------------------

		#------------ Save data --------------------------
		df_grp.to_hdf(file_data,key="df_grp")
		df_src.to_hdf(file_data,key="df_src")
		#-------------------------------------------------
	#-------------- End of loop for cases ---------------------------------------------------------------

	#----------- Append ----------------
	dfs_grp.append(df_grp)
	dfs_src.append(df_src)
	#------------------------------------

#------------ Concatenate --------------------
df_grp = pn.concat(dfs_grp,ignore_index=False)
df_src = pn.concat(dfs_src,ignore_index=False)
#---------------------------------------------

#------------ Save data --------------------------
df_grp.to_hdf(file_data_all,key="df_grp")
df_src.to_hdf(file_data_all,key="df_src")
#-------------------------------------------------

#=========================== Plots =======================================
if do_plt_grp_cmn:
	print("Plotting common group-level parameters")
	#------------ Read data --------------------------------
	df_grp = pn.read_hdf(file_data_all,key="df_grp")
	#-------------------------------------------------------

	#------------ Select only the n_stars = 100 group --------
	df_grp = df_grp.groupby("n_stars").get_group(100)
	df_grp.reset_index(inplace=True)
	#--------------------------------------------------------

	#------------- Split into all and linear models --------------------------
	mask_cmn = df_grp['Parameter'].str.contains("loc|std",regex=True)
	df = df_grp.loc[mask_cmn]
	#-------------------------------------------------------------------------

	pdf = PdfPages(filename=file_plt_grp_cmn)
	for st in sts_grp:
		fg = sns.FacetGrid(data=df,
						col="Parameter",
						sharey=False,
						sharex=True,
						margin_titles=True,
						col_wrap=3,
						hue="Case")
		
		fg.map(plt.fill_between,"distance",
				st["key"]+"_low",
				st["key"]+"_up",
				alpha=0.1)
		fg.map(sns.lineplot,"distance",st["key"]+"_mu")
		fg.add_legend()

		#------------ Labels ----------------------------
		axs = fg.axes_dict
		for par in df["Parameter"]:
			if par in ["loc[X]","loc[Y]","loc[Z]"]:
				axs[par].set_ylim(st["lim_loc_pos"])
			if par in ["loc[U]","loc[V]","loc[W]"]:
				axs[par].set_ylim(st["lim_loc_vel"])
			if par in ["std[X]","std[Y]","std[Z]"]:
				axs[par].set_ylim(st["lim_std_pos"])
			if par in ["std[U]","std[V]","std[W]"]:
				axs[par].set_ylim(st["lim_std_vel"])
			if "X" in par or "U" in par:
				axs[par].set_ylabel(st["name"])
			else:
				axs[par].set_yticklabels([])
			axs[par].set_xlabel("Distance [pc]")
			axs[par].title.set_text(par)
		#-----------------------------------------------

		sns.move_legend(fg,loc="lower center",
				bbox_to_anchor=(.45, 1), ncol=5)
		plt.subplots_adjust(wspace=0.1)
		pdf.savefig(bbox_inches='tight',dpi=300)
		plt.close()
	pdf.close()
	#-------------------------------------------------------------------------

if do_plt_grp_lnr:
	print("Plotting linear group-level parameters")
	#-------------------------- Read data --------------------------------
	dfs_grp = []
	for signal in [10,50,100]:
		file_tmp = dir_syn + "Gaussian_linear_{0}/data.h5".format(signal)
		tmp = pn.read_hdf(file_tmp,key="df_grp")
		tmp["C"] = signal
		dfs_grp.append(tmp)
	#----------------------------------------------------------------------

	df_grp = pn.concat(dfs_grp,ignore_index=False)
	df_grp.reset_index(level="Case",drop=True,inplace=True)
	
	#-------------- Compute SNR ----------------------------------------
	df_grp["SNR_mu"]  = df_grp.apply(lambda x: 100./x["unc_mu"],axis=1)
	df_grp["SNR_low"] = df_grp.apply(lambda x: 100./x["unc_up"],axis=1)
	df_grp["SNR_up"]  = df_grp.apply(lambda x: 100./x["unc_low"],axis=1)
	#--------------------------------------------------------------------

	#------------ Select specific groups -------------------------
	df_kap = df_grp.groupby("Parameter").get_group("kappa[X]")
	#------------------------------------------------------------

	df_kap.reset_index(inplace=True)
	df_grp.reset_index(inplace=True)

	#------------- Select linear parameter --------------------------
	mask_lnr = df_grp['Parameter'].str.contains("kappa|omega",regex=True)
	df = df_grp.loc[mask_lnr].copy()
	#-------------------------------------------------------------------------

	
	#----------------- SNR ---------------------------------
	g = sns.lineplot(data=df_kap,x="distance",y="SNR_mu",
						style="C",
						hue="n_stars",
						palette="tab10",
						style_order=[100,50,10],
						zorder=0)

	dfg = df_kap.groupby(["n_stars","C"])
	for name,tmp in dfg.__iter__():
		g.fill_between(
			x=tmp["distance"],
			y1=tmp["SNR_low"],
			y2=tmp["SNR_up"],
			color="grey",
			alpha=0.1)

	g.set(xlabel="Distance [pc]",ylabel="SNR")
	plt.savefig(file_plt_grp_lnr.replace(".pdf",".png"),bbox_inches='tight',dpi=300)
	plt.close()

	pdf = PdfPages(filename=file_plt_grp_lnr)

	#------------------- All parameters --------------------------
	for st in sts_grp:
		fg = sns.FacetGrid(data=df,
						row="Parameter",
						col="C",
						sharey=True,
						sharex=True,
						margin_titles=True,
						hue="n_stars")
		
		fg.map(plt.fill_between,"distance",
				st["key"]+"_low",
				st["key"]+"_up",
				alpha=0.1)
		fg.map(sns.lineplot,"distance",st["key"]+"_mu")
		fg.add_legend()

		#------------ Labels ----------------------------
		fg.set_axis_labels("Distance [pc]", st["name"])
		#-----------------------------------------------

		sns.move_legend(fg,loc="lower center",
				bbox_to_anchor=(.45, 1), ncol=5)
		plt.subplots_adjust(wspace=0.1)
		pdf.savefig(bbox_inches='tight',dpi=300)
		plt.close()
	pdf.close()


if do_plt_grp_spc:
	print("Plotting specific group-level parameters")
	#------------ Read data --------------------------------
	df_grp = pn.read_hdf(file_data_all,key="df_grp")
	#-------------------------------------------------------

	#------------ Select only the n_stars = 100 group --------
	df_grp = df_grp.groupby("n_stars").get_group(100)
	df_grp.reset_index(inplace=True)
	#--------------------------------------------------------

	#------------- Split into all and linear models --------------------------
	mask_cmn = df_grp['Parameter'].str.contains("weights|nu",regex=True)
	df = df_grp.loc[mask_cmn]
	#-------------------------------------------------------------------------

	pdf = PdfPages(filename=file_plt_grp_spc)
	for st in sts_grp:
		fg = sns.FacetGrid(data=df,
						col="Parameter",
						sharey=False,
						sharex=True,
						margin_titles=True,
						col_wrap=3,
						hue="Case")
		
		fg.map(plt.fill_between,"distance",
				st["key"]+"_low",
				st["key"]+"_up",
				alpha=0.1)
		fg.map(sns.lineplot,"distance",st["key"]+"_mu")
		fg.add_legend()

		#------------ Labels ----------------------------
		axs = fg.axes_dict
		for par in df["Parameter"]:
			axs[par].set_xlabel("Distance [pc]")
			if par == "nu":
				axs[par].set_ylabel(st["name"])
			axs[par].title.set_text(par)
		#-----------------------------------------------
		sns.move_legend(fg,loc="lower center",
				bbox_to_anchor=(.45, 1), ncol=5)

		pdf.savefig(bbox_inches='tight',dpi=300)
		plt.close()
	pdf.close()
	#-------------------------------------------------------------------------

if do_plt_src:
	print("Plotting source-level parameters")
	#------------ Read data --------------------------------
	df_src = pn.read_hdf(file_data_all,key="df_src")
	#-------------------------------------------------------

	
	#------------ Select only the n_stars = 100 group --------
	df = df_src.groupby("n_stars").get_group(100)
	df.reset_index(inplace=True)
	#--------------------------------------------------------

	#-------------- Source level----------------------------------------------
	pdf = PdfPages(filename=file_plt_src)
	for st in sts_src:
		fg = sns.FacetGrid(data=df,
						col="Parameter",
						sharey=False,
						sharex=True,
						margin_titles=True,
						col_wrap=3,
						hue="Case")
		fg.map(plt.fill_between,"distance",
				st["key"]+"_low",
				st["key"]+"_up",
				alpha=0.1)
		fg.map(sns.lineplot,"distance",st["key"]+"_mu")
		fg.add_legend()

		#------------ Labels ----------------------------
		axs = fg.axes_dict
		for par in df["Parameter"]:
			if par in coordinates[:3]:
				axs[par].set_ylim(st["lim_pos"])
			if par in coordinates[3:]:
				axs[par].set_ylim(st["lim_vel"])
			if "X" in par or "U" in par:
				axs[par].set_ylabel(st["name"])
			else:
				axs[par].set_yticklabels([])
			axs[par].set_xlabel("Distance [pc]")
			axs[par].title.set_text(par)
		#-----------------------------------------------
		sns.move_legend(fg,loc="lower center",
				bbox_to_anchor=(.45, 1), ncol=5)

		plt.subplots_adjust(wspace=0.1)

		pdf.savefig(bbox_inches='tight')
		plt.close()
	pdf.close()
	#-------------------------------------------------------------------------

if do_plt_time:
	#------------ Read data -------------------------------------
	df_time = pn.read_hdf(file_data_all,key="df_time")
	#------------------------------------------------------------

	#------------------------------------------------------------
	df_time.reset_index(drop=False,inplace=True)
	for tmp in ["Time","Time_low","Time_up"]:
		df_time[tmp] = df_time[tmp]/3600.
	#-----------------------------------------------------------

	fg = sns.FacetGrid(data=df_time,
					palette="viridis_r",
					hue="distance",
					legend_out=True,
					height=3,
					aspect=1)
	fg.map(sns.lineplot,"n_stars","Time")
	fg.map(plt.fill_between,"n_stars",
			"Time_low",
			"Time_up",
			alpha=0.2)
	fg.add_legend(title="Distance [pc]")
	fg.set(xlabel="N stars",ylabel="Time [hrs]")

	plt.savefig(file_plot_tme,bbox_inches='tight',dpi=200)
	plt.close()