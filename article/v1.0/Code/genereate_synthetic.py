import sys
import os
import numpy  as np
import pandas as pn
import scipy.stats as st
from time import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from kalkayotl.Transformations import phaseSpaceToAstrometry_and_RV
from kalkayotl.Functions import AngularSeparation,CovarianceParallax
from kalkayotl.Priors import eff,king

from pygaia.astrometry.vectorastrometry import cartesianToSpherical
from pygaia.errors.astrometric import parallaxErrorSkyAvg

from isochrones import get_ichrone
from isochrones.priors import ChabrierPrior

dir_main      =  "/home/javier/Repositories/Kalkayotl/Data/Synthetic/"
random_seeds  = [1,2,3,4,5,6,7,8,9,10]    # Random state for the synthetic data
n_stars       = 1000   # 100,500, 1000
metallicity   = 0.02  # Solar metallicity
age           = 8.2   # Typical value of Bossini+2019
mass_limit    = 4.0   # Avoids NaNs in photometry
labels        = ["ID","r","ra","dec","parallax","parallax_error","G"]
tracks        = get_ichrone('mist', tracks=True,bands="VIG")
extension_yrs = 0.0

####### FAMILY PARAMETERS ####################################################
family        = "GMM"
scale         = 10.0

#---- Only for GMM -------
scale2        = 20.0 # Second component
fraction      = 0.5  # Of first component
shift         = 0.1  # Relative distance shift of second comp.
#----- Only for EFF -----
gamma         = 3
# Only for King ---------
tidal_radius  = 5 # In units of core radius (i.e. scale)
#-----------------------------------------------------------

distances     = [100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000]
A             = np.eye(3)*scale
B             = np.eye(3)*scale2
###############################################################################

for seed in random_seeds:
	#------ Set Seed -----------------
	np.random.RandomState(seed=seed*int(time() % 100))
	#------ Directories and files --------------------------------
	dir_out      = dir_main  + family + "_" + str(n_stars) + "_" + str(seed)

	#------- Create directories -------
	os.makedirs(dir_out,exist_ok=True)

	#================= Loop over distances ==============================================
	print("Generating observables ...")
	for distance in distances:
		print(family +" at " + str(distance)+" pc")

		#------- Local files -----------------------------------------------------
		file_name    = dir_out   + "/{0}_{1}".format(family,str(int(distance)))
		file_plot    = file_name + ".pdf"
		file_data    = file_name + ".csv"


		#====================== Generate Astrometric Data ==================================================
		#------- Sample the radial distance  ------------------------------
		if family == "Uniform":
			r = st.uniform.rvs(loc=-1,scale=2,size=(n_stars,3))
		elif (family == "Gaussian") or (family == "GMM"):
			r = st.norm.rvs(size=(n_stars,3))
		elif family == "EFF":
			r = eff.rvs(gamma=gamma,size=3*n_stars).reshape(n_stars,3)
		elif family == "King":
			r = king.rvs(rt=tidal_radius,size=3*n_stars).reshape(n_stars,3)
		else:
			sys.exit("{0} family is not implemented".format(family))


		X = np.array([0.0,distance,0.0]) + np.matmul(r,A)

		if family == "GMM":
			n_stars_A = int(fraction*n_stars)
			n_stars_B = n_stars-n_stars_A
			X[-n_stars_B:] = np.array([0.0,distance*(1+shift),0.0]) + np.matmul(r[-n_stars_B:],B)
		#--------------------------------------------------------

		#------- Sky coordinates -----------
		r,ra,dec = cartesianToSpherical(X[:,0],X[:,1],X[:,2])

		#----- Transform ----------------------
		true_plx  = 1000.0/r             # In mas
		ra        = np.rad2deg(ra)       # In degrees
		dec       = np.rad2deg(dec)      # In degrees
		positions = np.column_stack((ra,dec))


		#------ Angular separations ----------
		theta = AngularSeparation(positions)

		#------ Covariance of spatial correlations -------------------
		cov_corr_plx = CovarianceParallax(theta,case="Vasiliev+2019")

		#=========================================================================================

		#============= Generate Photometric Data =================================================
		#------- Sample from Chbarier prior-------
		masses  = ChabrierPrior().sample(10*n_stars)

		#------- Only stars less massive than limit ------
		masses  = masses[np.where(masses < mass_limit)[0]]
		masses  = np.random.choice(masses,n_stars)
		df_phot = tracks.generate(masses, age, metallicity, distance=distance, AV=0.0)
		df_phot.dropna(inplace=True)
		#======================================================================================

		#-------Computes parallax error in mas ------------------------
		parallax_error = 1e-3*parallaxErrorSkyAvg(df_phot["G_mag"], 
								df_phot["V_mag"]-df_phot["I_mag"], 
								extension=extension_yrs)

		#------ Covariance of observational uncertainties --------
		cov_obs_plx = np.diag(parallax_error**2)

		#------ Total covariance is the convolution of the previous ones
		cov_plx =  cov_obs_plx + cov_corr_plx 

		#------- Correlated observations ------------------------------
		obs_plx = st.multivariate_normal.rvs(mean=true_plx,cov=cov_plx,size=1)
		#--------------------------------------------------------

		#========== Saves the synthetic data ====================
		data = pn.DataFrame(data={labels[0]:np.arange(n_stars),labels[1]:r,
					labels[2]:ra,labels[3]:dec,labels[4]:obs_plx,
					labels[5]:parallax_error,labels[6]:df_phot["G_mag"]})

		data.to_csv(path_or_buf=file_data,index_label=labels[0])
		#=====================================================

		#---------------- Plot ----------------------------------------------------
		pdf = PdfPages(filename=file_plot)
		n_bins = 100

		plt.figure(0)
		plt.scatter(ra,dec,s=1)
		plt.ylabel("Dec. [deg]")
		plt.xlabel("R.A. [deg]")
		pdf.savefig(bbox_inches='tight')
		plt.close(0)

		plt.figure(0)
		plt.hist(obs_plx,density=False,bins=n_bins,alpha=0.5,label="Observed")
		plt.hist(true_plx,density=False,bins=n_bins,alpha=0.5,label="True")
		if family == "GMM":
			plt.hist(true_plx[:n_stars_A],density=False,bins=n_bins,alpha=0.5,label="One")
			plt.hist(true_plx[n_stars_A:],density=False,bins=n_bins,alpha=0.5,label="Two")
		plt.legend()
		plt.ylabel("Density")
		plt.xlabel("Parallax [mas]")
		pdf.savefig(bbox_inches='tight')
		plt.close(0)

		plt.figure(0)
		plt.hist(parallax_error,density=False,bins=n_bins,log=True)
		plt.ylabel("Density")
		plt.xlabel("parallax_error [mas]")
		pdf.savefig(bbox_inches='tight')
		plt.close(0)

		plt.figure(0)
		plt.scatter(df_phot["V_mag"]-df_phot["I_mag"],df_phot["G_mag"],s=1)
		plt.ylabel("G [mag]")
		plt.xlabel("V - I [mag]")
		plt.ylim(25,3)
		pdf.savefig(bbox_inches='tight')
		plt.close(0)

		pdf.close()
	#------------------------------------------------------------------------------











