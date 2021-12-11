import sys
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as Gaussian
from sklearn.mixture import GaussianMixture
from pygaia.astrometry.vectorastrometry import phase_space_to_astrometry

#----------- Files ----------------
dir_main = "/home/jolivares/Cumulos/ComaBer/Kalkayotl/Fuernkranz+2019/6D_Gaussian_central/"

file_parameters = dir_main + "Cluster_statistics.csv"
file_original   = dir_main + "Synthetic_original.csv"
file_model      = dir_main + "Synthetic_model.csv"

n_stars = 10000
number_of_gaussians = [4]#[1,2,3,4,5,10,15]

param = pd.read_csv(file_parameters,usecols=["Parameter","mode"])

#---- Extract parameters ------------------------------------------------
loc  = param.loc[param["Parameter"].str.contains("loc"),"mode"].values
param.fillna(value=1.0,inplace=True)
stds = param.loc[param["Parameter"].str.contains('stds'),"mode"].values
corr = param.loc[param["Parameter"].str.contains('corr'),"mode"].values
#------------------------------------------------------------------------

#---- Construct covariance --------------
stds = np.diag(stds)
corr = np.reshape(corr,(6,6))
cov  = np.dot(stds,corr.dot(stds))
#-----------------------------------------

#--------- Generate synthetic samples ----------------------
phase_space = Gaussian(mean=loc,cov=cov).rvs(size=n_stars)
astrometry_rv  = np.array(phase_space_to_astrometry(
				phase_space[:,0],
				phase_space[:,1],
				phase_space[:,2],
				phase_space[:,3],
				phase_space[:,4],
				phase_space[:,5]
				))
astrometry_rv[0] = np.rad2deg(astrometry_rv[0])
astrometry_rv[1] = np.rad2deg(astrometry_rv[1])
#------------------------------------------------------------

#---------------- Fit ------------------
astrometry = astrometry_rv[:5].T
gmms = []
bics = []
aics = []
for ng in number_of_gaussians:
	print("--------")
	gmm = GaussianMixture(
		max_iter=1000,
		n_components=ng, 
		n_init=10,
		tol=1e-5)
				
	gmm.fit(astrometry)
	bics.append(gmm.bic(astrometry))
	aics.append(gmm.aic(astrometry))
	gmms.append(gmm)

# fig = plt.figure()
# plt.plot(aics,label="AIC")
# plt.plot(bics,label="BIC")
# plt.legend()
# plt.show()

best = gmms[np.argmin(bics)]

sample,_ = best.sample(n_samples=n_stars)


#---------- Data Frame original ---------------------
df_ori = pd.DataFrame(data={
				"ra":astrometry_rv[0],
				"dec":astrometry_rv[1],
				"parallax":astrometry_rv[2],
				"pmra":astrometry_rv[3],
				"pmdec":astrometry_rv[4],
				"radial_velocity":astrometry_rv[5]
				})
df_ori.to_csv(file_original,index_label="source_id")
#---------------------------------------------------

#---------- Data Frame model ---------------------
df_mod = pd.DataFrame(data={
				"ra":sample[:,0],
				"dec":sample[:,1],
				"parallax":sample[:,2],
				"pmra":sample[:,3],
				"pmdec":sample[:,4]
				})
df_mod.to_csv(file_model,index_label="source_id")
#------------------------------------------------

#----- Save model -------

#- Likelihood computation must include uncertainties
#-- So it is best to do it on the GPUs





