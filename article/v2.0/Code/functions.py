import sys
import os
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from astropy.table import Table


def weighted_mean(means,stds):
	variances = stds**2
	weights  = 1./variances
	weighted_variance = 1/np.sum(weights)
	mu = weighted_variance*np.sum(means*weights)
	sd = np.sqrt(weighted_variance)
	return mu, sd


def clean_members(velocity_model,maha,
	file_in,
	file_out,
	coordinates = ["U","V","W"],
	columns = ["source_id","label"]
	):
	
	for c in coordinates:
		columns.append("mean_"+c)

	#---------- Read members ---------------------------
	df_in = pd.read_csv(file_in,usecols=columns)
	df_in.set_index(["source_id","label"],inplace=True)
	#---------------------------------------------------

	print("Sources in the input file: {0}".format(df_in.shape[0]))

	#----------- Drop field and label ---------------------
	df_in.drop(index="Field",level="label",
				inplace=True,errors="ignore")
	#------------------------------------------------------

	#------------- Loop over MCD -----------------------------
	if velocity_model == "linear":
		mcd = MinCovDet(random_state=0).fit(
			df_in.loc[:,["mean_"+c for c in coordinates]])
		df_in["maha"] = np.sqrt(mcd.dist_)
		mask = df_in["maha"]> maha
		cond = any(mask)
		drop = df_in.loc[mask].index
		df_in.drop(index=drop,inplace=True)
	#--------------------------------------------------------

	#------- Save ---------
	df_in.to_csv(file_out)
	#----------------------

	print("Sources in the output file: {0}".format(df_in.shape[0]))

def process_radial_velocities(case,
	file_gaia,
	file_apogee,
	file_ids=None,
	gaia_radial_velocity_name="dr3_radial_velocity",
	):
	
	#----------- APOGEE ---------------------------------------------------------------
	apogee_columns = ["RA","DEC","GAIAEDR3_SOURCE_ID","VHELIO_AVG","VSCATTER","VERR"]
	apogee_rename = {"VHELIO_AVG":"apogee_radial_velocity","GAIAEDR3_SOURCE_ID":"source_id"}
	#----------------------------------------------------------------------------------

	#-------------------- Gaia ---------------------------
	gaia_mapper = {gaia_radial_velocity_name:"gaia_radial_velocity"}
	for k,v in gaia_mapper.copy().items():
		gaia_mapper[k+"_error"] = v+"_error"
	#-----------------------------------------------------

	#============= Load members =========================
	#----- Load catalogue ------------------------
	if '.csv' in file_gaia:
		df_cat = pd.read_csv(file_gaia)
	elif ".fits" in file_gaia:
		dat = Table.read(file_gaia, format='fits')
		df_cat  = dat.to_pandas()
		del dat
	else:
		sys.exit("Format file not recognized. Only CSV of FITS")

	df_cat.rename(columns=gaia_mapper,inplace=True)
	df_cat.set_index("source_id",inplace=True)
	#-------------------------------------------------------------

	if (file_ids is not None) and os.path.exists(file_ids):
		print("Merging with members")
		#-------------- Read ids ----------------------------------
		df_ids = pd.read_csv(file_ids, usecols=["source_id"])
		df_ids.set_index("source_id",inplace=True)
		#-------------------------------------------------------------

		#---- Combine catalogue with members --------
		df = df_cat.merge(df_ids,how="inner",
					left_index=True,
					right_index=True)
		#----------------------------------------
	else:
		print("WARNING: Outliers not removed!!")
		df = df_cat

	#--- Assert that observed values have uncertainties and viceversa ----
	nan_rvs = np.isnan(df["gaia_radial_velocity"].values)
	nan_unc = np.isnan(df["gaia_radial_velocity_error"].values)
	np.testing.assert_array_equal(nan_rvs,nan_unc,
	err_msg="Gaia: There are discrepant radial velocity missing uncertainties and values!")
	#---------------------------------------------------------------------
	surveys = ["gaia"]
	#==============================================================

	if case == "original":
		print("The radial velocities will be left untouched!")

	else:
		if "A" in case:
			#=============== APOGEE ===============================
			print("Merging with APOGEE ...")
			#----- Load APOGEE ----------------------------------
			apogee = Table.read(file_apogee, format='fits',hdu=1)
			#----------------------------------------------------

			#--- Extract desired columns ----------------
			apogee = apogee[apogee_columns]
			#--------------------------------------------

			#- Transform to pandas DF ---
			apogee = apogee.to_pandas()
			#----------------------------

			#------------- RV error ---------------
			apogee["apogee_radial_velocity_error"] = np.where(
				apogee["VSCATTER"] == 0.0, 
				apogee["VERR"],
				apogee["VSCATTER"])
			#--------------------------------------

			#------- Rename columns ---------------------------
			apogee.rename(columns=apogee_rename,inplace=True)
			#--------------------------------------------------

			#------ Drop missing RA,DEC ----------------------------
			apogee.dropna(subset=["RA","DEC"],inplace=True)
			#-------------------------------------------------------

			#--------- Drop unused columns ----------------------
			apogee = apogee.loc[:,[
			"source_id",
			"apogee_radial_velocity",
			"apogee_radial_velocity_error"]]
			#----------------------------------------------------

			#----------------------- Set index ----------------------
			apogee.drop_duplicates(subset="source_id",inplace=True)
			apogee.set_index("source_id",inplace=True)
			#---------------------------------------------------------

			#----------------- Merge ---------------------
			df = df.merge(apogee,how="left",
						left_index=True,right_index=True,
						sort=False)
			#---------------------------------------------

			#----------- Fix missing incongruency --------------------
			for a,b in zip(["","_error"],["_error",""]):
				df["apogee_radial_velocity"+a] = df.apply(lambda x: 
					x["apogee_radial_velocity"+a] if \
					np.isfinite(x["apogee_radial_velocity"+b]) else \
					np.nan,axis=1)
			#-------------------------------------------------------------

			#--- Assert that observed values have uncertainties and viceversa --------------------------
			nan_rvs = np.isfinite(df["apogee_radial_velocity"].values)
			nan_unc = np.isfinite(df["apogee_radial_velocity_error"].values)
			np.testing.assert_array_equal(nan_rvs,nan_unc,
			err_msg="APOGEE: There are discrepant radial velocity missing uncertainties and values!")
			#------------------------------------------------------------------------------------------

			surveys.append("apogee")
			#================================================================

		if "S" in case:
			#=============== Simbad X-Match =================================
			print("Merging with Simbad ...")

			df.reset_index(inplace=True)
			#----------- Query by name -----------------------------------
			df["Name"] = df.apply(lambda x: "Gaia DR3 {0}".format(
									np.int_(x["source_id"])),axis=1)

			tb_simbad = Simbad.query_objects(df["Name"],verbose=False)
			assert tb_simbad is not None, "Simbad: Empty query result!"
			df_simbad = tb_simbad.to_pandas()
			#-------------------------------------------------------------

			#---------- Drop redshift values ---------------------------------
			df_simbad.drop(index=df_simbad[df_simbad["RVZ_TYPE"] != "v"].index,
							inplace=True)
			#------------------------------------------------------------------

			#---------- Drop rows with no rv uncertainty ----------------
			df_simbad.dropna(how="any",subset=["RVZ_RADVEL","RVZ_ERROR"],
							inplace=True)
			#-------------------------------------------------------------

			#------------- Rename simbad columns -----------------------
			df_simbad.rename(columns={
						"RVZ_RADVEL":"simbad_radial_velocity",
						"RVZ_ERROR":"simbad_radial_velocity_error"},
						inplace=True)
			#-----------------------------------------------------------

			#--- Assert that observed values have uncertainties and viceversa ----
			nan_rvs = np.isnan(df_simbad["simbad_radial_velocity"].values)
			nan_unc = np.isnan(df_simbad["simbad_radial_velocity_error"].values)
			np.testing.assert_array_equal(nan_rvs,nan_unc,
			err_msg="Simbad: There are discrepant radial velocity missing uncertainties and values!")
			#---------------------------------------------------------------------

			#---------- Set index as the original query number ---
			df_simbad.set_index("SCRIPT_NUMBER_ID",inplace=True)
			#----------------------------------------------------

			#------------------- Merge by original query number -----------------
			df = df.merge(df_simbad,left_index=True,right_index=True,how="left")
			df.set_index("source_id",inplace=True)
			#-------------------------------------------------------------------

			surveys.append("simbad")
			#================================================================

		print("Processing merged catalogues ...")
		assert df.index.is_unique, "Index values are not unique. Remove duplicated sources!"

		#----------- Use APOGEE or Gaia or Simbad in that order------------
		df["final_radial_velocity"] = np.nan
		df["final_radial_velocity_error"] = np.nan
		df["final_radial_velocity_origin"] = "--"

		for survey in ["simbad","gaia","apogee"]:
			if survey in surveys:
				for suffix in ["","_error"]:
					df["final_radial_velocity"+suffix] = df.apply(lambda x:
						x[survey+"_radial_velocity"+suffix] if \
						np.isfinite(x[survey+"_radial_velocity"]) else \
						x["final_radial_velocity"+suffix],
						axis=1)
				df["final_radial_velocity_origin"] = df.apply(lambda x:
						survey if np.isfinite(x[survey+"_radial_velocity"]) else \
						x["final_radial_velocity_origin"],
						axis=1)
		#---------------------------------------------------------------------

	return df

def filter_members(df,file_out,
	args = {
		"rv_error_limits":[0.01,20.0], # It need revision possibly extending it
		"ruwe_threshold":1.4,
		"radial_velocity_sd_clipping":2.0,
		"parallax_sd_clipping":2.0,
		"allow_rv_missing":True},
	radial_velocity_name="final_radial_velocity",
	parallax_limits={"min":5.0,"max":np.inf}
	):

	#--------------- Mappers -----------------------------------------
	output_mapper = {"radial_velocity":radial_velocity_name}
	base_error = "{0}_error"
	for k,v in output_mapper.copy().items():
		output_mapper[base_error.format(k)] = base_error.format(v)

	input_mapper  = { v:k for k,v in output_mapper.items()}
	df.rename(columns=input_mapper,inplace=True)
	#------------------------------------------------------------------

	#-------- Drop sources faraway sources ------
	mask_parallax = (df["parallax"] >= parallax_limits["min"]) &\
		(df["parallax"] <= parallax_limits["max"])
	print("Number of sources with valid parallax: {0}".format(
		sum(mask_parallax)))
	df = df.loc[mask_parallax,:].copy()
	#--------------------------------------------


	print("Replacing minumum and maximum uncertainties ...")
	#----------- Set minimum uncertainty -------------------------------------
	condition = df["radial_velocity_error"] < args["rv_error_limits"][0]
	df.loc[condition,"radial_velocity_error"] = np.float32(args["rv_error_limits"][0])
	#-------------------------------------------------------------------------

	#----------- Set maximum uncertainty -------------------------------------
	condition = df["radial_velocity_error"] > args["rv_error_limits"][1]
	df.loc[condition,"radial_velocity"] = np.nan
	df.loc[condition,"radial_velocity_error"]  = np.nan
	#-------------------------------------------------------------------------

	#------------- Binaries -------------------------------
	condition = df.loc[:,"ruwe"] > args["ruwe_threshold"]
	df.loc[condition,"radial_velocity"] = np.nan
	df.loc[condition,"radial_velocity_error"]  = np.nan
	print("Binaries: {0}".format(sum(condition)))
	#-----------------------------------------------------

	#---------- Outliers --------------------------------------------
	for var in ["parallax","radial_velocity"]:#,"radial_velocity","radial_velocity"]:
		mu = np.nanmean(df[var])
		sd = np.nanstd(df[var])
		print("{0}: {1:2.1f} +/- {2:2.1f}".format(var,mu,sd))
		maha_dst = np.abs(df[var] - mu)/sd
		condition = maha_dst > args[var+"_sd_clipping"]
		df.loc[condition,var] = np.nan
		df.loc[condition,var+"_error"]  = np.nan
		print("Outliers in {0}: {1}".format(var,sum(condition)))
	#----------------------------------------------------------------

	#---------------- Drop parallax outliers ---------------------
	df.dropna(subset=["parallax","parallax_error"],inplace=True)
	#-------------------------------------------------------------

	#------------------------ Allow RV missing ------------------------
	if not args["allow_rv_missing"]:
		df.dropna(subset=["radial_velocity","radial_velocity_error"],
			inplace=True)
	#------------------------------------------------------------------

	# df.rename(columns=output_mapper,inplace=True)

	print("Saving file ...")
	#------- Save as csv ---------
	df.to_csv(file_out)
	#-----------------------------

	print("Final number of members: {0}".format(df.shape[0]))
	#==================================================================


def infer_model(prior,
	file_data,
	dir_out,
	nuts_sampler="numpyro",
	chains=2,
	cores=2,
	tuning_iters=4000,
	sample_iters=2000,
	init_iters=int(1e5),
	init_refine=False,
	prior_predictive=True,
	target_accept=0.65,
	sky_error_factor=1e6
	):

	#================= Inference and Analysis ====================
	kal = Inference(dimension=6,
					dir_out=dir_out,
					zero_points={
						"ra":0.,
						"dec":0.,
						"parallax":-0.017,
						"pmra":0.,
						"pmdec":0.,
						"radial_velocity":0.},
					indep_measures=False,
					reference_system="Galactic",
					sampling_space="physical")

	kal.load_data(file_data,sky_error_factor=sky_error_factor)

	kal.setup(prior=prior["type"],
			  parameters=prior["parameters"],
			  hyper_parameters=prior["hyper_parameters"],
			  parameterization=prior["parameterization"])

	kal.plot_pgm()

	kal.run(sample_iters=sample_iters,
			tuning_iters=tuning_iters,
			target_accept=target_accept,
			chains=chains,
			cores=cores,
			init_iters=init_iters,
			init_refine=init_refine,
			step_size=None,
			nuts_sampler=nuts_sampler,
			prior_predictive=prior_predictive,
			prior_iters=chains*sample_iters)

	kal.load_trace()
	kal.convergence()
	kal.plot_chains()
	kal.plot_prior_check()
	kal.plot_model()
		# groups_kwargs={
		# "color":{"A":"tab:blue","B":"tab:orange"},
		# "mapper":{"A":"Halo","B":"Core"}},
		# ticks={"minor":16,"major":5},)
	kal.save_statistics()
	# kal.save_samples()
	#=========================================================


def trace_orbit(df,age,
	observables_names = ["ra","dec","parallax","pmra","pmdec","radial_velocity"]):
	# import sys
	# import numpy as np
	# import pandas as pd
	# import scipy.stats as st

	# import matplotlib
	# # matplotlib.use('Agg')
	# import matplotlib.pyplot as plt
	# import matplotlib.cm as cm
	# import matplotlib.lines as mlines
	# from matplotlib.colors import Normalize
	# from matplotlib.collections import LineCollection
	# import matplotlib.ticker as ticker
	# import seaborn as sb
	# import arviz as az
	# import h5py

	from Functions_trace_back import Obs2Phs#,get_samples,my_mode
	# from Functions_trace_back import compute_cov,compute_mcd,compute_std

	from astropy import units as u
	# from astropy.coordinates import SkyCoord, ICRS, Galactocentric

	from galpy.orbit import Orbit
	from galpy.potential import MWPotential2014

	ts = np.linspace(0.,age,int(np.abs(age)*2))*u.Myr

	#--------- Read ------------------------------------------------
	df_src = df.loc[:,[obs+"_true" for obs in observables_names]]
	#---------------------------------------------------------------

	#---------------------------- Distance ------------------------------------
	df_src["distance_[kpc]"] = df_src.apply(lambda x: 1./x["parallax_true"],axis=1)
	#--------------------------------------------------------------------------

	#------------------ Extract only Galpy columns ------------
	df_src = df_src.loc[:,
	["ra_true","dec_true","distance_[kpc]",
	"pmra_true","pmdec_true","radial_velocity_true"]]
	#----------------------------------------------------------

	#-------------------- Obrit ----------------
	O_src = Orbit(df_src.to_numpy(),
					radec=True,
					ro=8.,
					vo=220.,
					solarmotion="schoenrich")
	O_src.turn_physical_on()
	O_src.integrate(ts,MWPotential2014)
	#-------------------------------------------

	#-------- Back to DataFrame ------------------------------------------
	df_fwd = pd.DataFrame(data={
		# "X_gal":O_src.x(t,use_physical=True).flatten()*1.e3,
		# "Y_gal":O_src.y(t,use_physical=True).flatten()*1.e3,
		# "Z_gal":O_src.z(t,use_physical=True).flatten()*1.e3,
		# "U_gal":O_src.vx(t,use_physical=True).flatten(),
		# "V_gal":O_src.vy(t,use_physical=True).flatten(),
		# "W_gal":O_src.vz(t,use_physical=True).flatten(),
		"X_helio":O_src.helioX(ts[-1],use_physical=True).flatten()*1.e3,
		"Y_helio":O_src.helioY(ts[-1],use_physical=True).flatten()*1.e3,
		"Z_helio":O_src.helioZ(ts[-1],use_physical=True).flatten()*1.e3,
		"U_helio":O_src.U(ts[-1],use_physical=True).flatten(),
		"V_helio":O_src.V(ts[-1],use_physical=True).flatten(),
		"W_helio":O_src.W(ts[-1],use_physical=True).flatten(),
		},index=df.index)
	#--------------------------------------------------------------------

	#----------------------- Transform --------------------------
	o2p = Obs2Phs(transformation="ICRS2GAL",solar_motion=None)
	smp_fwd = o2p.backward(df_fwd.to_numpy())
	#------------------------------------------------------------

	#--------- DataFrame----------------------------------------
	df_fwd = pd.DataFrame(data=smp_fwd,
				columns=[obs+"_true" for obs in observables_names],
				index=df_fwd.index)
	#-------------------------------------------------------------

	# pd.testing.assert_frame_equal(df,df_fwd,rtol=1e-5)
	# sys.exit()

	return df_fwd

	