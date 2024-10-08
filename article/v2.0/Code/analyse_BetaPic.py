#------------ LOAD LIBRARIES -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import numpy as np
import pandas as pn

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import dill
import h5py
import arviz as az

sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 2.5})

family = "Gaussian"
dimension = "6D"

#---------------------- Directories and data -------------------------------
dir_main  = "/home/jolivares/Projects/Kalkayotl/BetaPic/"
dir_plots = "/home/jolivares/Dropbox/MisArticulos/Kalkayotl/Figures/"
dir_tabs  = "/home/jolivares/Dropbox/MisArticulos/Kalkayotl/Tables/"
dir_run   = "/6D_Gaussian_Galactic_joint_1E+06/"
authors   = ["Miret-Roig+2020","Couture+2023","Crundall+2019"]
files_src = ["Table3.csv","core_sources.csv",None]
file_data_all = dir_main  + "Data.h5"

file_plot_src = dir_plots + "BetaPic_source-level.png"
file_plot_grp = dir_plots + "BetaPic_group-level.png"
file_plot_cnv = dir_plots + "BetaPic_convergence.png"
file_plot_age = dir_plots + "BetaPic_age.png"
file_tab_grp  = dir_tabs  + "BetaPic_group-level.tex"

do_all_dta = True
do_plt_cnv = False
do_plt_grp = False
do_tab_grp = True
do_plt_src = False
do_plt_age = False

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

def remove_6D(x):
	return x.replace("6D::","")

def format_mu_sd(mu,sd):
	if np.isnan(sd):
		return "${0:2.2f}$".format(mu)
	else:
		return "${0:2.2f}\\pm{1:2.2f}$".format(mu,sd)


columns_fmt = [
			("loc[X]","[pc]"),
			("loc[Y]","[pc]"),
			("loc[Z]","[pc]"),
			("loc[U]","[km/s]"),
			("loc[V]","[km/s]"),
			("loc[W]","[km/s]"),
			("std[X]","[pc]"),
			("std[Y]","[pc]"),
			("std[Z]","[pc]"),
			("std[U]","[km/s]"),
			("std[V]","[km/s]"),
			("std[W]","[km/s]")]
#-----------------------------------------------------------------------

#------------ Literature values --------------------
data_lit ={"Crundall+2019":{"6D::loc[X]":[3.1,30.0,3.2],
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
		   "Miret-Roig+2020":{
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
			"Couture+2023":{
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
	dfs_src = []
	for author,file in zip(authors,files_src):

		#------------- Files ----------------------------------
		dir_chains = dir_main + author + dir_run
		file_grp   = dir_chains  + "Cluster_statistics.csv"
		file_src   = dir_chains  + "Sources_statistics.csv"
		#-----------------------------------------------------

		if file is not None:
			#================= Literature sources ==========================================
			#----------------- Real file if it exists ----------------------
			tmp_lit = pn.read_csv("{0}{1}/{2}".format(dir_main,author,file),usecols=obs_src_columns)
			tmp_inf = pn.read_csv(file_src, usecols=obs_src_columns)
			
			tmp_inf.set_index("source_id",inplace=True)
			tmp_lit.set_index("source_id",inplace=True)

			#--------------- Join into coordinates --------------------------
			tmps = []
			for dat in [tmp_inf,tmp_lit]:
				tmp_crd = []
				for coord in coordinates:		
					tmp = dat.loc[:,["mean_"+coord,"sd_"+coord]]
					tmp.rename(columns={"mean_"+coord:"mean","sd_"+coord:"sd"},
							inplace=True)
					tmp.set_index(pn.MultiIndex.from_product(
							[[author],[coord],tmp.index.values]),
							inplace=True)
					tmp.index.set_names(["Authors","Coordinate","source_id"],
						inplace=True)
					tmp_crd.append(tmp)
				tmp_crd = pn.concat(tmp_crd,ignore_index=False)
				tmps.append(tmp_crd)
			#------------------------------------------------------------------

			#----------------- Merge -------------
			df_src = tmps[0].merge(tmps[1],
				left_index=True,right_index=True,
				suffixes=("_inf","_lit"))
			#-------------------------------------

			dfs_src.append(df_src)

	df_src = pn.concat(dfs_src,ignore_index=False)
	#==================================================================

	#================== Inferred parameters ===================================
	dfs_grp = []
	dfs_lit = []
	for author in authors:
		#------------- Files ----------------------------------
		dir_chains = dir_main + author + dir_run
		file_grp   = dir_chains  + "Cluster_statistics.csv"
		file_src   = dir_chains  + "Sources_statistics.csv"
		#------------------------------------------------------

		#---------------- Group level ----------------------------
		tmp = pn.DataFrame.from_dict(data=data_lit[author],
									orient="index",
									columns=["low","mean","up"])
		if author == "Crundall+2019":
			tmp.loc[parameters[:6],"mean"] -= LSR

		tmp["sd"] = tmp.apply(lambda x: 0.5*(x["low"]+x["up"]),axis=1)
		tmp.set_index(pn.MultiIndex.from_product(
					[[author],tmp.index.values]),
					# names=["Authors","Parameter"],
					inplace=True)
		dfs_lit.append(tmp.reset_index())
		#--------------------------------------------------------

		#---------------- Read parameters ----------------------------
		df_grp = pn.read_csv(file_grp,usecols=obs_grp_columns)
		df_grp.set_index("Parameter",inplace=True)
		df_grp.set_index(pn.MultiIndex.from_product(
			[[author],df_grp.index.values]),
			# names=["Authors","Parameter"],
			inplace=True)
		#-------------------------------------------------------------

		#------------ Relative error ----------------
		df_grp["true"] = tmp.loc[:,"mean"]
		df_grp["agreement"] = df_grp.apply(lambda x:(x["true"] > x["hdi_2.5%"]) and
								 (x["true"]<x["hdi_97.5%"]),axis=1)
		#----------------------------------------------------------------------------------

		#------------ Lower and upper limits --------------------------------
		df_grp["lower"] = df_grp.apply(lambda x:x["mean"]-x["hdi_2.5%"],axis=1)
		df_grp["upper"] = df_grp.apply(lambda x:x["hdi_97.5%"]-x["mean"],axis=1)
		df_grp["low"]   = df_grp["sd"]
		df_grp["up"]    = df_grp["sd"]
		#--------------------------------------------------------------------

		#----------- Append ----------------
		dfs_grp.append(df_grp.reset_index())
		#------------------------------------

	#------------ Concatenate --------------------
	df_grp = pn.concat(dfs_grp,ignore_index=True)
	df_lit = pn.concat(dfs_lit,ignore_index=True)
	#--------------------------------------------
	#=================================================================================

	#----------- Set indices -----------------------------
	df_grp.set_index(["level_0","level_1"],inplace=True)
	df_lit.set_index(["level_0","level_1"],inplace=True)
	df_grp.index.set_names(["Authors","Parameter"],inplace=True)
	df_lit.index.set_names(["Authors","Parameter"],inplace=True)
	#----------------------------------------------------------

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

if do_tab_grp:
	df_grp = df_grp.loc[:,["mean","sd","agreement"]]
	df_lit = df_lit.loc[:,["mean","sd"]]

	df_lit.replace(0.0,value=np.nan,inplace=True)
	df_grp["Origin"] = "This work"
	df_lit["Origin"] = "Reported"

	tab = pn.concat([df_lit,df_grp],ignore_index=False)
	tab.reset_index(inplace=True)
	tab["string"] = tab.apply(lambda x: format_mu_sd(x["mean"],x["sd"]),axis=1)

	# tab.drop(columns=["mean","sd"],inplace=True)
	tab.set_index(["Parameter"],inplace=True)
	tab = tab.pivot(columns=["Authors","Origin"],values=["string","agreement"])
	tab.sort_index(axis=1,level=0,inplace=True)
	tab.rename(axis=0,mapper=remove_6D,inplace=True)
	tab = tab.loc[[col[0] for col in columns_fmt]]

	print(tab.loc[:,("agreement",slice(None),"This work")])

	tab = tab.loc[:,"string"]
	tab.index = pn.MultiIndex.from_tuples(columns_fmt)
	tab.index.names = ["Parameter","Units"]
	tab.columns.names = [None,None]

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
	

if do_plt_cnv:
	#-------------- Convergence ----------------------------------------------
	pdf = PdfPages(filename=file_plot_cnv)
	for st in sts_cnv:
		fg = sns.FacetGrid(data=df_grp.reset_index(),
						col="Parameter",
						sharey=False,
						margin_titles=True,
						col_wrap=3,)
		fg.map(sns.scatterplot,"Authors",st["key"])
		fg.add_legend()
		fg.set_axis_labels("Authors",st["name"])
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
					margin_titles=False,
					col_wrap=3,
					height=2,
					aspect=1.5,
					)
	fg.map(sns.scatterplot,"mean","Authors",
			zorder=2,color="tab:red")
	fg.set_axis_labels("Value","")
	axs = fg.axes_dict
	dfg_lit = df_lit.groupby("Parameter")
	dfg_grp = df_grp.groupby("Parameter")
	for parameter in parameters:
		ax = axs[parameter]

		#----------- Literature values -----------------------
		dfl = dfg_lit.get_group(parameter)
		ax.errorbar(x=dfl["mean"],y=dfl["Authors"],
					xerr=dfl.loc[:,["low","up"]].to_numpy().T,
					capsize=5,elinewidth=1,capthick=1,
					fmt="none",color="tab:red",zorder=0)
		#----------------------------------------------------

		#------ Inferred HDI values -------------------------------
		dfg = dfg_grp.get_group(parameter)
		ax.errorbar(x=dfg["mean"],y=dfg["Authors"],
					xerr=dfg.loc[:,["low","up"]].to_numpy().T,
					capsize=0,elinewidth=1,
					fmt="none",
					ecolor="tab:grey",
					zorder=0)
		#------------------------------------------------------

		#------ Inferred sd values -------------------------------
		dfg = dfg_grp.get_group(parameter)
		ax.errorbar(x=dfg["mean"],y=dfg["Authors"],
					xerr=dfg["sd"],
					capsize=5,elinewidth=1,capthick=1,
					fmt=".",
					ms=15,
					barsabove=False,
					color="tab:olive",
					ecolor="tab:grey",
					zorder=1)
		#------------------------------------------------------
	
	plt.savefig(file_plot_grp,bbox_inches='tight')
	plt.close()
	#-------------------------------------------------------------------------


if do_plt_src:

	def annotate_rms(data, **kws):
		rms = np.sqrt(np.mean((data["mean_inf"]-data["mean_lit"])**2))
		ax = plt.gca()
		y = 0.05 if kws["label"]=="Miret-Roig+2020" else 0.15
		ax.text(.6, y,"RMS : {0:2.2f}".format(rms), 
			transform=ax.transAxes,
			fontweight="normal",**kws)

	df_src.reset_index(level=["Authors","Coordinate"],inplace=True)

	#-------------- Source level----------------------------------------------
	fg = sns.FacetGrid(data=df_src,
					col="Coordinate",
					hue="Authors",
					sharey=False,
					sharex=False,
					margin_titles=True,
					col_wrap=3)
	fg.map(sns.scatterplot,"mean_lit","mean_inf")
	fg.set_axis_labels("Literature","This work")
	fg.add_legend()
	sns.move_legend(
    fg, "lower center",
    bbox_to_anchor=(.4, 0.98), 
    ncol=2, 
    frameon=False)

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

	plt.savefig(file_plot_src,bbox_inches='tight',dpi=300)
	plt.close()
	#-------------------------------------------------------------------------

if do_plt_age:
	def age(kappa):
		return  1./(1.022712165*kappa)

	#------------- Read posterior samples of Txx and Tyyy -------------------------------------
	file_h5 = dir_main +"Miret-Roig+2020" +"/" + "6D_Gaussian_Galactic_linear_1E+06/Samples.h5"
	with h5py.File(file_h5,'r') as hf:
		kappa = np.array(hf.get("Cluster/6D::kappa"))
	#------------------------------------------------------------------------------------------

	#----------- DF kappa -------------------------------------------------------------
	df_kappa = pn.DataFrame(data=kappa,columns=["X","Y","Z"])
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




