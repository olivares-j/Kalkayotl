import sys
import numpy as np
import pandas as pd
import h5py

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pygaia.astrometry.vectorastrometry import phase_space_to_astrometry

#----------- Files ----------------
dir_main = "/home/jolivares/Cumulos/ComaBer/Kalkayotl/Fuernkranz+2019/fully_observed/test/"

file_members = dir_main + "members+rvs.csv"
file_samples = dir_main + "6D_1Gaussian_central/Samples.h5"
file_scatter = dir_main + "Scatter_predicted_vs_observed.png"
file_histogr = dir_main + "Histogram_observed-predicted.png"

n_samples = 1000

df_obs = pd.read_csv(file_members,usecols=["source_id","observed_radial_velocity","observed_radial_velocity_error"])
grp_keys = df_obs["source_id"].values


#=============== Kalkayotl =========================================
with h5py.File(file_samples,'r') as hf:
	#---------------- Sources --------------------------------------
	srcs = hf.get("Sources")
	samples = np.empty((len(grp_keys),n_samples,6))

	#-------- loop over array and fill it with samples -------
	for i,ID in enumerate(grp_keys):
		#--- Extracts a random choice of the samples ------
		tmp = np.array(srcs.get(str(ID)))
		idx = np.random.choice(np.arange(tmp.shape[1]),
								size=n_samples,
								replace=False)
		samples[i] = tmp[:,idx].T
	#---------------------------------------------------
samples = samples.reshape((-1,6))

ra,dec,parallax,mua,mud,rv = phase_space_to_astrometry(
							samples[:,0],
							samples[:,1],
							samples[:,2],
							samples[:,3],
							samples[:,4],
							samples[:,5])

#--------- Samples DF -------------------------------------
df = pd.DataFrame(data=rv,
			columns=["predicted_radial_velocity"],
			index=pd.MultiIndex.from_product(
				iterables=[grp_keys,np.arange(n_samples)],
				names=["source_id","sample"])
				)
#---------------------------------------------------------

dfm = df.groupby(level=0).mean()
dfe = df.groupby(level=0).std()
df_pre = dfm.join(dfe,rsuffix="_error")
df = df_obs.join(df_pre,on="source_id")

fig = plt.figure(figsize=None)
ax = plt.gca()
#--------- Sources --------------------------
ax.errorbar(x=df["observed_radial_velocity"],
			y=df["predicted_radial_velocity"],
			xerr=df["observed_radial_velocity_error"],
			yerr=df["predicted_radial_velocity_error"],
			fmt='none',
			ecolor="grey",
			elinewidth=1,
			zorder=1)
ax.scatter(x=df["observed_radial_velocity"],
		   y=df["predicted_radial_velocity"],
		   color="blue",s=3,zorder=2)
ax.plot(ax.get_xlim(),ax.get_xlim(),ls="--",color="black",lw=1,zorder=0)
#------------- Titles -------------------------------------
ax.set_xlabel("Observed radial velocity [km/s]")
ax.set_ylabel("Predicted radial velocity [km/s]")
ax.set_xlim(-15,15)
ax.set_ylim(-3,3)
plt.tight_layout()
plt.savefig(file_scatter,dpi=300)
plt.close()

plt.hist(df["observed_radial_velocity"]-df["predicted_radial_velocity"],bins=25)
plt.xlabel("observed - predicted [km/s]")
plt.tight_layout()
plt.savefig(file_histogr,dpi=300)
plt.close()