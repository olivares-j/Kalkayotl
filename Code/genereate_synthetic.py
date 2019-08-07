import os
import numpy  as np
import pandas as pn
import scipy.stats as st
from Transformations import phaseSpaceToAstrometry

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


'''
TAP QUERY at
http://gea.esac.esa.int/tap-server/tap

SELECT TOP 10000000 ra_error,dec_error,parallax_error,pmra_error,pmdec_error,radial_velocity_error
FROM gaiadr2.gaia_source
order by random_index

'''

#----------------Mock data and MCMC parameters  --------------------
case          = "Gauss_2"
random_state  = 1234     # Random state for the synthetic data
mu            = np.array([96,-277,-86,7,-26,-48],dtype="float32")
sd_0          = np.array([3,8,6,4,14,4],dtype="float32")
sd_1          = 2.0*sd_0
fraction      = np.array([0.8,0.2])
n_stars       = 1000         # Number of mock distances
labels        = ["ID","r","x","y","z","vx","vy","vz",
				"ra","dec","parallax","pmra","pmdec","radial_velocity",
				"ra_error","dec_error","parallax_error","pmra_error",
				"pmdec_error","radial_velocity_error"]
#--------------------------------------------------------------------------------------

#------ Directories and files --------------------------------
dir_main  = os.getcwd()[:-4]
dir_data  = dir_main  + "Data/"
file_data = dir_data  + "Gaia_DR2_uncertainty.csv"
dir_case  = dir_data  + case +"/"
file_plot = dir_case  + case + "_plot.pdf"
file_syn  = dir_case  + case + ".csv"
#-----------------------------------------------------


#------- Create directories -------
os.makedirs(dir_case,exist_ok=True)

#----- Reads observed uncertainties --------
u_obs = pn.read_csv(file_data)

#----- Drop missing values ----------
u_obs.dropna(how='any',inplace=True)
u_obs = u_obs.to_numpy()

#----- Choose only the needed uncertainties --------
u_obs = u_obs[np.random.choice(len(u_obs),n_stars)]

#====================== Generate Synthetic Data ==================================================
#---------- True stars--------------------------------------------------------
n_0  = int(np.ceil(fraction[0]*n_stars))
n_1  = int(np.floor(fraction[1]*n_stars))

true = st.multivariate_normal.rvs(mean=mu, cov=np.diag(sd_0**2), 
						size=n_0,random_state=random_state)
if n_1 > 0:
	true_1  = st.multivariate_normal.rvs(mean=mu, cov=np.diag(sd_1**2), 
						size=n_1,random_state=random_state)
	true = np.vstack((true,true_1))

N,D = np.shape(true)
if N != n_stars:
	sys.exit("Incorrect number of objects")

#---- Physical to observed quantities-------
dist  = np.sqrt(true[:,0]**2 + true[:,1]**2 + true[:,2]**2)

#--- Notice that  phaseSpace... returns a theano tensor---
true = np.array(phaseSpaceToAstrometry(true).eval())
#----------------------------------------------------

#------- Observed ------------------------------
print("Generating observables ...")
obs = np.zeros_like(true)
for i in range(N):
	for j in range(D):
		obs[i,j] = st.norm.rvs(loc=true[i,j],scale=u_obs[i,j],size=1)
#--------------------------------------------------------

#========== Saves the synthetic data ====================
data = np.column_stack((dist,true,obs,u_obs))
df = pn.DataFrame(data=data,columns=labels[1:])
df.to_csv(path_or_buf=file_syn,index_label=labels[0])
#=====================================================

#---------------- Plot ----------------------------------------------------
n_bins = 100
pdf = PdfPages(filename=file_plot)
plt.hist(obs[:,2],density=False,bins=n_bins,alpha=0.5,label="Observed")
plt.hist(1000.0/dist,density=False,bins=n_bins,alpha=0.5,label="True")
plt.legend()
plt.ylabel("Density")
plt.xlabel("Parallax [mas]")
pdf.savefig(bbox_inches='tight')
plt.close()
pdf.close()
#------------------------------------------------------------------------------











