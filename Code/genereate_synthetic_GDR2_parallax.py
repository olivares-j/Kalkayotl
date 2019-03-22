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
random_state        = 1234     # Random state for the synthetic data
data_loc,data_scale = 300.,20.0   # Location and scale of the distribution for the mock data
data_distribution   = st.norm  # Change it according to your needs
name_synthetic      = "Gaussian_300_20"
synthetic_uncert    = "random"    # The type of synthetic uncertainties: "random" or "linear"
n_stars             = 1000         # Number of mock distances
labels              = ["ID","dist","parallax","parallax_error"]
#--------------------------------------------------------------------------------------

#--------------------- Directories and files --------------------------------
dir_main  = os.getcwd()[:-4]
dir_data  = dir_main  + "Data/"
dir_ana   = dir_main  + "Analysis/"
dir_plots = dir_ana   + "Plots/"
file_data = dir_data  + "Gaia_DR2_parallax_uncertainty.csv"
file_plot = dir_plots + name_synthetic + "_uncertainty.pdf"
file_syn  = dir_data  + name_synthetic + ".csv"

#------- Create directories -------
if not os.path.isdir(dir_ana):
	os.mkdir(dir_ana )
if not os.path.isdir(dir_plots):
    os.mkdir(dir_plots)
#---------------------------------

#---------- Reads observed uncertainties --------
u_obs = np.array(pn.read_csv(file_data))

#====================== Generate Synthetic Data ============================================================================
#---------- create synthetic data -------------------------------------------------------------------------
dst      = data_distribution.rvs(loc=data_loc, scale=data_scale, size=n_stars,random_state=random_state)
#---- obtains the parallax in mas -------
pax      = (1.0/dst)*1e3
#----- assigns an uncertainty similar to those present in Gaia DR2 data, in mas ------- 
if synthetic_uncert is "random":
	#----------------- Fit --------------------
	factor = 10. # This factor improves the fit
	param = st.chi2.fit(u_obs*factor)
	u_syn = st.chi2.rvs(df=param[0],loc=param[1], scale=param[2], size=n_stars,random_state=random_state)/factor
elif synthetic_uncert is "linear":
	u_syn    = np.linspace(np.min(u_obs),np.max(u_obs), num=n_stars)
else:
	sys.exit("Incorrect synthetic_uncert type")
#----------------------------------------------------

#========== Saves the synthetic data ====================
data = np.column_stack((dst,pax,u_syn))
df = pn.DataFrame(data=data,columns=labels[1:])
df.to_csv(path_or_buf=file_syn,index_label=labels[0])
#=====================================================

#---------------- Plot ----------------------------------------------------
n_bins = 100
pdf = PdfPages(filename=file_plot)
plt.hist(u_obs,density=True,bins=n_bins,alpha=0.5,label="Observed")
plt.hist(u_syn,density=True,bins=n_bins,alpha=0.5,label="Synthetic")
plt.legend()
plt.ylabel("Density")
plt.xlabel("Parallax uncertainties [mas]")
pdf.savefig(bbox_inches='tight')
plt.close()
pdf.close()
#------------------------------------------------------------------------------











