import sys
import os
import numpy as np
import dill
os.environ["ISOCHRONES"] = "/raid/jromero/isochrones/"

list_of_distances = [100.,200.,800.,1600.]
list_of_n_stars   = [100,200,400]
list_of_seeds     = [0,1,2,3,4]
true_pos_sds      = np.array([9.,9.,9.])
true_vel_loc      = np.array([10.,10.,10.])
true_vel_sds      = np.array([1.,1.,1.])
model             = "Gaussian"
velocity_model    = "linear"

astrometric_args = {
			"position":{"family":model,
						"location":np.array([0.0,0.0,0.0]),
						"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":model,
						"location":true_vel_loc,
						"covariance":np.diag(true_vel_sds**2)
						}
					}

if velocity_model == "linear":
	astrometric_args["velocity"]["kappa"] = np.ones(3)
	astrometric_args["velocity"]["omega"] = np.array([
											[-1,-1,-1],
											[1,1,1]
											])
photometric_args = {
"log_age": 8.0,    
"metallicity":0.012,
"Av": 0.0,         
"mass_limits":[0.01,4.5], 
"bands":["G","BP","RP"],
"mass_prior":"Uniform"
}

dill.dump_session("./globals_{0}.pkl".format(model))

dir_repos = "/home/jromero/Repos"
dir_main  = "/raid/jromero/Kalkayotl/Synthetic/Gaussian_{0}/".format(velocity_model)

#----- Amasijo -------------------
path_amasijo   = dir_repos + "/Amasijo/"
sys.path.append(path_amasijo)
from Amasijo import Amasijo
#---------------------------------


for distance in list_of_distances:
	for n_stars in list_of_n_stars:
		for seed in list_of_seeds:
			print(20*"-")
			base_name = "{0}_n{1}_d{2}_s{3}".format(
				model,int(n_stars),int(distance),seed)
			print(base_name)

			if os.path.isfile(dir_main+base_name+".csv"):
				continue

			astrometric_args["position"]["location"] = np.array([distance,0.0,0.0])
			
			ama = Amasijo(
						astrometric_args=astrometric_args,
						photometric_args=photometric_args,
						reference_system="Galactic",
						label_radial_velocity="radial_velocity",
						seed=seed)

			ama.generate_cluster(file=dir_main+base_name+".csv",
								n_stars=n_stars,
								angular_correlations=None)

			ama.plot_cluster(file_plot=dir_main+base_name+".pdf")