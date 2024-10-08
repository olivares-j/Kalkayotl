import sys
import os
import numpy as np
import pandas as pn
import dill

from groups import *

os.environ["ISOCHRONES"] = dir_iso

case = "Gaussian_linear"

dir_base  = dir_syn + "{0}_100/".format(case)
os.makedirs(dir_base,exist_ok=True)
dill.dump_session("{0}/globals.pkl".format(dir_base))

#----- Amasijo -------------------
sys.path.append(dir_ama)
from Amasijo import Amasijo
#---------------------------------

for n_stars in list_of_n_stars:
	for distance in list_of_distances:
		case_args = CASE_ARGS["{0}_d{1}".format(case,int(distance))]
		for seed in list_of_seeds:
			tmp = "n{0}_d{1}_s{2}/".format(int(n_stars),int(distance),seed)
			print(40*"-" +" " + tmp + " " +40*"-")
			dir_tmp = dir_base + tmp
			os.makedirs(dir_tmp,exist_ok=True)

			file_mem = dir_tmp + "synthetic.csv"
			file_plt = dir_tmp + "synthetic.pdf"

			if os.path.isfile(file_mem):
				continue
			
			ama = Amasijo(
						phasespace_args=case_args,
						isochrones_args=case_args["isochrones_args"],
						photometry={
							"labels":{
								"phot_g_mean_mag":"phot_g_mean_mag",
								"phot_bp_mean_mag":"phot_bp_mean_mag",
								"phot_rp_mean_mag":"phot_rp_mean_mag"},
							"family":"Gaia"},
						radial_velocity={
							"labels":{"radial_velocity":"radial_velocity"},
							"family":"Gaia"},
						reference_system="Galactic",
						seed=seed)

			ama.generate_cluster(file=file_mem,
								n_stars=n_stars,
								angular_correlations=None,
								max_mahalanobis_distance=3.0
								)

			ama.plot_cluster(file_plot=file_plt)
