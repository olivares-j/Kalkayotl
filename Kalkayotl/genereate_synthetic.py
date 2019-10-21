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
random_state     = 12345     # Random state for the synthetic data
dist_name        = "Gaussian"
scl              = 20.0
#----- Only if King profile-----
rt    = 20.0
#----- Only if EFF profile -----
gamma = 3.0

#--------- Stars ----------------
n_stars          = None
labels           = ["ID","r","parallax","parallax_error",
				    "ra","dec"]
#--------------------------------------------------------------------------------------

list_of_cases =[
{"name":"Pleiades",    "loc":100},
{"name":"Pleiades",    "loc":200},
{"name":"Ruprecht_147","loc":300},
{"name":"Ruprecht_147","loc":400},
{"name":"Ruprecht_147","loc":500},
{"name":"NGC_1647",    "loc":600},
{"name":"NGC_2264",    "loc":700},
{"name":"NGC_2264",    "loc":800},
{"name":"NGC_2682",    "loc":900},
{"name":"NGC_2682",    "loc":1000},
{"name":"NGC_2244",    "loc":1500},
{"name":"NGC_188",    "loc":2000},
{"name":"IC_1848",     "loc":2500},
{"name":"IC_1848",     "loc":3000},
{"name":"IC_1848",     "loc":3500},
{"name":"IC_1848",     "loc":4000},
{"name":"NGC_6791",     "loc":4500},
{"name":"NGC_6791",     "loc":5000},
{"name":"NGC_3603",     "loc":10000},
]

#------ Directories and files --------------------------------
dir_main     = os.getcwd()[:-10] + "/Data/"
dir_out      = dir_main  + "Synthetic/" + dist_name + "_" + str(int(scl))

#------- Create directories -------
os.makedirs(dir_out,exist_ok=True)


#================= Loop over cases ==============================================
print("Generating observables ...")
for case in list_of_cases:
	print(30*"=")
	print(case["name"])

	#------- Local files -----------------------------------------------------
	file_data    = dir_main  + case['name'] + ".csv"
	file_name    = dir_out   + "/{0}_{1}".format(dist_name,str(int(case["loc"])))
	file_plot    = file_name + ".pdf"
	file_syn     = file_name + ".csv"

	#---------- Select distribution --------------------------
	if dist_name is "Gaussian":
		distribution     = st.norm(loc=case["loc"],scale=scl)
	elif dist_name is "EFF":
		distribution     = eff(r0=case["loc"],rc=scl,gamma=gamma)
	elif dist_name is "King":
		distribution     = king(r0=case["loc"],rc=scl,rt=rt)
	else:
		sys.exit("Distribution not recognized")


	#====================== Read true cluster data ===================================================
	data = pn.read_csv(file_data,usecols=labels[3:],nrows=n_stars)
	data.dropna(inplace=True)
	#---------- True stars--------------------------------------------------------
	n_stars = data.shape[0]

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











