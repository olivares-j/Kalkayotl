import sys
import numpy as np

#----- Amasijo -------------------
path_amasijo   = "/home/jolivares/Repos/Amasijo/"
sys.path.append(path_amasijo)
from Amasijo import Amasijo
#---------------------------------

dir_main  = "/home/jolivares/Repos/Kalkayotl/article/v2.0/Synthetic/"

list_of_distances = [100.,200.,400.,800.,1600.]
list_of_n_stars   = [100,200,400]
list_of_seeds     = [0,1,2,3,4]
model             = "Gaussian"

photometric_args = {
"log_age": 8.0,    
"metallicity":0.012,
"Av": 0.0,         
"mass_limits":[0.01,4.5], 
"bands":["G","BP","RP"],
"mass_prior":"Uniform"
}

for distance in list_of_distances:
	for n_stars in list_of_n_stars:
		for seed in list_of_seeds:
			astrometric_args = {
			"position":{"family":model,
						"location":np.array([distance,0.0,0.0]),
						"covariance":np.diag([9.,9.,9.])},
			"velocity":{"family":model,
						"location":np.array([-10.0,-10.0,-10.0]),
						"covariance":np.diag([1.,1.,1.])}}

			base_name = "{0}_n{1}_d{2}".format(model,n_stars,int(distance))
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