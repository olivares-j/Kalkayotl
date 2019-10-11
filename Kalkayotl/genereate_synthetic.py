import sys
import os
import numpy  as np
import pandas as pn
import scipy.stats as st
from Transformations import phaseSpaceToAstrometry_and_RV

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from EFF import eff
from King import king


#----------------Mock data and MCMC parameters  --------------------
case             = "IC_1848"
random_state     = 12345     # Random state for the synthetic data
loc              = 2700.0
scl              = 150.0
rt               = 20
dist_name        = "norm"
distribution     = st.norm(loc=loc,scale=scl)
# distribution     = eff(r0=loc,rc=scl,gamma=3.0)
# distribution     = king(r0=loc,rc=scl,rt=rt)
n_stars          = None
labels           = ["ID","r","parallax","parallax_error",
				    "ra","dec"]
#--------------------------------------------------------------------------------------

#------ Directories and files --------------------------------
dir_main     = os.getcwd()[:-4]
dir_out      = dir_main  + "Data/Synthetic/" + case
file_data    = dir_main  + "Data/"+case+".csv"
file_name    = dir_out   + "/{0}_{1}".format(case,str(int(loc)))
file_plot    = file_name + ".pdf"
file_syn     = file_name + ".csv"

#------- Create directories -------
os.makedirs(dir_out,exist_ok=True)

#====================== Read true cluster data ===================================================
data = pn.read_csv(file_data,usecols=labels[3:],nrows=n_stars)
data.dropna(inplace=True)
#---------- True stars--------------------------------------------------------
n_stars = data.shape[0]


print("Generating observables ...")
#====================== Generate Synthetic Data ==================================================
dist     = distribution.rvs(size=n_stars,random_state=random_state)
true_plx = 1000.0/dist
#----------------------------------------------------

#=================== Uncertainties =================================
#------- Observed ------------------------------
obs_plx = np.zeros_like(true_plx)
for i,(true_loc,true_scl) in enumerate(zip(true_plx,data["parallax_error"])):
	obs_plx[i] = st.norm.rvs(loc=true_loc,scale=true_scl,size=1)
#--------------------------------------------------------

#========== Saves the synthetic data ====================
df = data.copy(deep=True)
df.insert(loc=0,column="r",value=dist)
df.insert(loc=1,column="parallax",value=obs_plx)
df.to_csv(path_or_buf=file_syn,index_label=labels[0])
#=====================================================

#---------------- Plot ----------------------------------------------------
pdf = PdfPages(filename=file_plot)
n_bins = 100
plt.hist(obs_plx,density=False,bins=n_bins,alpha=0.5,label="Observed")
plt.hist(true_plx,density=False,bins=n_bins,alpha=0.5,label="True")
plt.legend()
plt.ylabel("Density")
plt.xlabel("Parallax [mas]")
pdf.savefig(bbox_inches='tight')
plt.close()
pdf.close()
#------------------------------------------------------------------------------











