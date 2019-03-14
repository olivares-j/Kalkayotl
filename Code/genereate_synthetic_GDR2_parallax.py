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
data_loc,data_scale = 300,20   # Location and scale of the distribution for the mock data
data_distribution   = st.norm  # Change it according to your needs
name_synthetic      = "Gaussian_300_20.csv"
n_stars             = 1000         # Number of mock distances
labels              = ["ID","dist","parallax","parallax_error"]
#--------------------------------------------------------------------------------------

#--------------------- Directories and files --------------------------------
dir_main  = os.getcwd()[:-4]
dir_data  = dir_main  + "Data/"
dir_ana   = dir_main  + "Analysis/"
dir_plots = dir_ana   + "Plots/"
file_data = dir_data  + "Gaia_DR2_parallax_uncertainty.csv"
file_plot = dir_plots + "Gaia_DR2_parallax_uncertainty.pdf"
file_syn  = dir_data  + name_synthetic

#------- Create directories -------
if not os.path.isdir(dir_ana):
	os.mkdir(dir_ana )
if not os.path.isdir(dir_plots):
    os.mkdir(dir_plots)

#----------------- Fit --------------------
u = np.array(pn.read_csv(file_data))*10.
param = st.chi2.fit(u)

#---------------- Plot ----------------------------------------------------
x = np.arange(min(u),max(u),(max(u)-min(u))/100)
pdf = PdfPages(filename=file_plot)
plt.hist(u,density=True,bins=100,label="Data")
pdf_fitted = st.chi2.pdf(x,param[0], loc=param[1], scale=param[2])
plt.plot(x,pdf_fitted,label="Fit")
plt.legend()
plt.xlabel("Parallax uncertainties [10 * mas]")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()
pdf.close()
#------------------------------------------------------------------------------


#====================== Generate Synthetic Data ============================================================================
#---------- create synthetic data -------------------------------------------------------------------------
dst      = data_distribution.rvs(loc=data_loc, scale=data_scale, size=n_stars,random_state=random_state)
#---- obtains the parallax -------
pax      = 1.0/dst
#----- assigns an uncertainty similar to those present in Gaia DR2 data ------- 
u_pax    = st.chi2.rvs(df=param[0],loc=param[1], scale=param[2], size=n_stars,random_state=random_state)
#----------------------------------------------------

#============================ Saves the synthetic data ===========================================================
#---------- Parallax in mas ------------------------------------------------------------------------------------
data = np.column_stack((dst,pax*1e3,u_pax/10.)) # Notice the factor 10 because of the fit to the uncertainties.
df = pn.DataFrame(data=data,columns=labels[1:])
df.to_csv(path_or_buf=file_syn,index_label=labels[0])








