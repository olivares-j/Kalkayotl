#------------ Load libraries -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#============ Directory and data ===========================================
dir_base = "/home/jolivares/Repos/Kalkayotl/article/v2.0/ComaBer/Core/"
#============================================================================

#=============== Tuning knobs ============================
dimension = 6
reference_system = "Galactic"
radec_precision_arcsec = 1.0
velocity_model = "joint"
nuts_sampler = "numpyro"
#=========================================================================================

#======================= Inference and Analysis =====================================================
dfs = []
for target_accept in [0.95,0.8,0.7,0.6,0.5]:
	#------ Output directories for each prior -------------------
	dir_prior = dir_base +  "{0}D_{1}_{2}_{3}_{4:1.0f}arcsec_{5}".format(
		dimension,
		"Gaussian",
		"central",
		nuts_sampler,
		radec_precision_arcsec,
		target_accept
		)
	#------------------------------------------------------------

	df = pd.read_csv(dir_prior+"/Cluster_statistics.csv")
	df["target_accept"] = target_accept

	mask = df["Parameter"].str.contains("0, 0") | \
		   df["Parameter"].str.contains("1, 1") | \
		   df["Parameter"].str.contains("2, 2") | \
		   df["Parameter"].str.contains("3, 3") | \
		   df["Parameter"].str.contains("4, 4") | \
		   df["Parameter"].str.contains("5, 5")

	df.drop(index=df.loc[mask].index,inplace=True)
	dfs.append(df)

df = pd.concat(dfs,ignore_index=True)
# df.set_index(["Parameter","target_accept"],inplace=True)

def case_assignement(x):
	if "loc" in x: 
		return "loc" 
	elif "std" in x:
		return "std"
	else:
		return "rho"

df["type"] = df["Parameter"].apply(case_assignement)

plt.figure(figsize=(30, 10))
fg = sns.FacetGrid(data=df,row="type",margin_titles=True)
fg.map(sns.scatterplot,"target_accept","r_hat")
fg.tight_layout()
fg.savefig(dir_base + "Comparison_target_accept_vs_rhat.png",figsize=(10,10))
plt.close()

fg = sns.FacetGrid(data=df,row="type",sharey=False,margin_titles=True)
fg.map(sns.scatterplot,"target_accept","ess_bulk")
fg.savefig(dir_base + "Comparison_target_accept_vs_ess_bulk.png")

dfg = df.groupby("type")

for par in ["loc","std"]:
	tmp = dfg.get_group(par)
	fg = sns.FacetGrid(data=tmp,col="Parameter",col_wrap=3,sharey=False,margin_titles=True)
	fg.map(sns.scatterplot,"target_accept","mean")
	fg.map(plt.errorbar, "target_accept","mean","sd", marker="o")
	fg.savefig(dir_base + "Comparison_target_accept_vs_mean_{0}.png".format(par))






