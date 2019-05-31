import os
import numpy  as np
import pandas as pn
import scipy.stats as st
import scipy.optimize as optimization

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

'''
TAP QUERY at
http://gea.esac.esa.int/tap-server/tap

SELECT TOP 1000000 parallax_error 
FROM gaiadr2.gaia_source
order by random_index

'''

#----------------Mock data and MCMC parameters  --------------------
case                = "Cluster_300"
random_state        = 1234     # Random state for the synthetic data
data_loc,data_scale = 300.,20.0   # Location and scale of the distribution for the mock data
data_distribution   = st.norm  # Change it according to your needs
name                = case + "_"+str(int(data_scale))
type_uncert         = "random"    # The type of synthetic uncertainties: "random" or "linear"
n_stars             = 1000         # Number of mock distances
labels              = ["ID","dist","parallax","parallax_error"]
#--------------------------------------------------------------------------------------

#--------------------- Directories and files --------------------------------
dir_main  = os.getcwd()[:-4]
dir_data  = dir_main  + "Data/"
file_data = dir_data  + "Gaia_DR2_parallax_uncertainty.csv"
dir_case  = dir_data  + case +"/"
file_plot = dir_case  + name + "_" + type_uncert + "_plot.pdf"
file_syn  = dir_case  + name + "_" + type_uncert +".csv"


#------- Create directories -------
if not os.path.isdir(dir_case):
	os.mkdir(dir_case)

#---------- Reads observed uncertainties --------
u_obs = np.array(pn.read_csv(file_data)).flatten()

#====================== Generate Synthetic Data ============================================================================
print("Fitting Gaia parallax uncertainty ---")
#---------- create synthetic distances -------------------------------------------------------------------------
distance  = data_distribution.rvs(loc=data_loc, scale=data_scale, size=n_stars,random_state=random_state)

#---- True parallax in mas -------
true_plx      = (1.0/distance)*1e3

#----- assigns an uncertainty similar to those present in Gaia DR2 data, in mas ------- 
if type_uncert is "random":
	#----------------- Random choice --------------------
	u_syn    = np.random.choice(u_obs,n_stars)
elif type_uncert is "linear":
	u_syn    = np.linspace(np.min(u_obs),np.max(u_obs), num=n_stars)
else:
	sys.exit("Incorrect type_uncert")
#----------------------------------------------------

#------- Observed parallax ------------------------------
print("Generating synthetic parallax ---")
obs_plx = np.zeros_like(true_plx)
for i,(tplx,uplx) in enumerate(zip(true_plx,u_syn)):
	obs_plx[i] = st.norm.rvs(loc=tplx,scale=uplx,size=1)
#--------------------------------------------------------

#========== Saves the synthetic data ====================
data = np.column_stack((distance,obs_plx,u_syn))
df = pn.DataFrame(data=data,columns=labels[1:])
df.to_csv(path_or_buf=file_syn,index_label=labels[0])
#=====================================================

#---------------- Plot ----------------------------------------------------
n_bins = 100
pdf = PdfPages(filename=file_plot)
plt.hist(obs_plx,density=False,bins=n_bins,alpha=0.5,label="Observed")
plt.hist(true_plx,density=False,bins=n_bins,alpha=0.5,label="True")
plt.legend()
plt.ylabel("Density")
plt.xlabel("Parallax [mas]")
pdf.savefig(bbox_inches='tight')
plt.close()
pdf.close()
#------------------------------------------------------------------------------











