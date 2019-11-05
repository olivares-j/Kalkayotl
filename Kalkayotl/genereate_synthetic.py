import sys
import os
import numpy  as np
import pandas as pn
import scipy.stats as st
from Transformations import phaseSpaceToAstrometry_and_RV

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Functions import AngularSeparation,CovarianceParallax

from EFF import eff
from King import king


#----------------Mock data and MCMC parameters  --------------------
random_seed  = 12345    # Random state for the synthetic data
dist_name    = "Gaussian"
labels       = ["ID","r","parallax","parallax_error","ra","dec"]
#--------------------------------------------------------------------------------------

list_of_cases =[
{"name":"Pleiades",    "loc":100 ,"scl":3},
{"name":"Pleiades",    "loc":200 ,"scl":3},
{"name":"Ruprecht_147","loc":300 ,"scl":8},
{"name":"Ruprecht_147","loc":400 ,"scl":8},
{"name":"Ruprecht_147","loc":500 ,"scl":8},
{"name":"NGC_1647",    "loc":600 ,"scl":17},
{"name":"NGC_2264",    "loc":700 ,"scl":30},
{"name":"NGC_2264",    "loc":800 ,"scl":30},
{"name":"NGC_2682",    "loc":900 ,"scl":15},
{"name":"NGC_2682",    "loc":1000,"scl":15},
{"name":"NGC_2244",    "loc":1600,"scl":220},
{"name":"NGC_188",     "loc":1800,"scl":60},
# {"name":"NGC_7789",    "loc":2000,"scl":},
# {"name":"IC_1848",     "loc":2300,"scl":160}
# {"name":"NGC_6791",    "loc":4500,"scl":400},
# {"name":"NGC_3603",    "loc":5800,"scl":900},
]

#------ Directories and files --------------------------------
dir_main     = os.getcwd() + "/Data/"
dir_out      = dir_main  + "Synthetic/" + dist_name + "_" + str(random_seed)

#------- Create directories -------
os.makedirs(dir_out,exist_ok=True)


#================= Loop over cases ==============================================
print("Generating observables ...")
for case in list_of_cases:
	print(30*"=")
	print(case["name"]+" at "+str(case["loc"])+" pc")

	#------- Local files -----------------------------------------------------
	file_data    = dir_main  + case['name'] + ".csv"
	file_name    = dir_out   + "/{0}_{1}".format(dist_name,str(int(case["loc"])))
	file_plot    = file_name + ".pdf"
	file_syn     = file_name + ".csv"

	#---------- Select distribution --------------------------
	if dist_name is "Gaussian":
		distribution     = st.norm(loc=case["loc"],scale=case["scl"])
	elif dist_name is "EFF":
		distribution     = eff(r0=case["loc"],rc=case["scl"],gamma=case["gamma"])
	elif dist_name is "King":
		distribution     = king(r0=case["loc"],rc=case["scl"],rt=case["rt"])
	else:
		sys.exit("Distribution not recognized")


	random_state = random_seed

	#====================== Read true cluster data ===================================================
	data = pn.read_csv(file_data,usecols=labels[3:])
	data.dropna(inplace=True)

	#---------- Number of stars --------------------------------------------------------
	n_stars = data.shape[0]

	#------ Obtain array of positions ------------
	positions = data[["ra","dec"]].to_numpy()

	#------ Angular separations ----------t
	theta = AngularSeparation(positions)

	#------ Covariance of spatial correlations -------------------
	cov_corr_plx = CovarianceParallax(theta,case="Vasiliev+2018")

	#------ Covariance of observational uncertainties --------
	cov_obs_plx = np.diag(data["parallax_error"]**2)

	#------ Total covariance is the convolution of the previous ones
	cov_plx =  cov_obs_plx + cov_corr_plx 

	#====================== Generate Synthetic Data ==================================================
	#-------- True values ---------------------------------------
	dist     = distribution.rvs(size=n_stars,random_state=random_state)
	true_plx = 1000.0/dist # distance to parallax in mas
	#----------------------------------------------------

	#------- Observed ------------------------------
	obs_plx = st.multivariate_normal.rvs(mean=true_plx,cov=cov_plx,size=1,
		random_state=random_state)
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











