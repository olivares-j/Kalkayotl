import sys
import dynesty
import numpy as np
import pandas as pn
import scipy.stats as st
from scipy.special import logsumexp,hyp2f1,gamma as gamma_function

from dynesty import plotting as dyplot

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from kalkayotl.Transformations import Iden,pc2mas
from kalkayotl.Priors import eff,king


################################## Evidence 1D ####################################
class Evidence1D():
	'''
	Infer evidence of a model
	'''
	def __init__(self,mu_data,sg_data,
		prior="Gaussian",
		parameters={"location":None,"scale": None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		N_samples=None,
		M_samples=10000,
		transformation="mas",
		quantiles=[0.05,0.5,0.95],
		seed=12345):
		np.random.seed(seed)

		self.M    = M_samples
		self.logM = np.log(M_samples)

		self.quantiles = quantiles

		if parameters["location"] is not None or parameters["scale"] is not None:
			sys.exit("This modules works only when location and scale are set free")

		#========================= Data ==================================================
		#-------- Assume independence amongst sources ------------------
		data = np.column_stack((mu_data,np.sqrt(np.diag(sg_data))))
		
		#----- Use all data or a random sample of it --------
		if N_samples == None:
			self.N    = len(mu_data)
			self.data = data
		else:
			self.N    = N_samples
			idx       = np.random.choice(np.arange(0,len(mu_data),1),self.N)
			self.data = data[idx]

		if self.N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")
		#================================================================================

		#============= Transformations ====================================

		if transformation == "mas":
			self.Transformation = Iden

		elif transformation == "pc":
			self.Transformation = pc2mas

		else:
			sys.exit("Transformation is not accepted")
		#==================================================================


		#================ Hyper-prior =====================================
		hp_loc = st.norm(loc=hyper_alpha[0],scale=hyper_alpha[1])
		hp_scl = st.gamma(a=2.0,scale=hyper_beta[0]/2.0)
		#========================================================================

		#================= Prior ========================================================
		#--------- Cluster oriented prior-----------------------------------------------
		if prior == "Uniform":
			self.D = 2
			self.names = ["loc","scl"]

			def prior_sample(theta):
				result = theta[0] + theta[1]*np.random.uniform(-1.0,1.0,size=self.M)
				return result

			def hp_transform(u):
				x = np.zeros_like(u)
				x[0] = hp_loc.ppf(u[0])
				x[1] = hp_scl.ppf(u[1])
				return x

		elif prior == "Gaussian":
			self.D = 2
			self.names = ["loc","scl"]

			def prior_sample(theta):
				result = theta[0] + theta[1]*st.norm.rvs(size=self.M)
				return result

			def hp_transform(u):
				x = np.zeros_like(u)
				x[0] = hp_loc.ppf(u[0])
				x[1] = hp_scl.ppf(u[1])
				return x

		elif prior == "GMM":

			if parameters["weights"] is None:

				# Components in the mixture
				G      = len(hyper_delta)

				self.D     = G*3
				self.names = sum([["weights_{0}".format(i) for i in range(G)],["loc_{0}".format(i) for i in range(G)],["scl_{0}".format(i) for i in range(G)]],[])

				def prior_sample(theta):
					samples = st.norm.rvs(size=(self.M,G))

					frc = theta[:G]
					loc = theta[G:(2*G)]
					scl = theta[(2*G):]

					result = np.dot(loc + scl*samples,frc)
					return result

				def hp_transform(u):
					# Transform unit cube hyper prior
					x = np.zeros_like(u)
					x[:G]      = u[:G]
					x[G:(2*G)] = hp_loc.ppf(u[G:(2*G)])
					x[(2*G):]  = hp_scl.ppf(u[(2*G):])

					x[:G] = x[:G]/np.sum(x[:G]) 
					return x

			else:
				G = len(hyper_delta)

				self.D     = 2*G
				self.names = sum([["loc_{0}".format(i) for i in range(G)],["scl_{0}".format(i) for i in range(G)]],[])

				def prior_sample(theta):
					samples = st.norm.rvs(size=(self.M,G))

					frc = np.array(parameters["weights"])
					loc = theta[:G]
					scl = theta[G:(2*G)]

					result = np.dot(loc + scl*samples,frc)
					return result

				def hp_transform(u):
					# Transform unit cube hyper prior
					x = np.zeros_like(u)
					x[:G]      = hp_loc.ppf(u[:G])
					x[G:(2*G)] = hp_scl.ppf(u[G:(2*G)])
					return x

		elif prior == "EFF":

			if parameters["gamma"] is None:
				self.D = 3
				self.names = ["loc","scl","gamma"]

				if hyper_gamma[0] < 2.0:
					sys.exit("Setting hyper_gamma[0] to values < 2.0 leads to extremely inefficient sampling")

				a, b = (2.0 - hyper_gamma[0]) / hyper_gamma[1], (10. - hyper_gamma[0]) / hyper_gamma[1]

				hp_gamma = st.truncnorm(a=a,b=b,loc=hyper_gamma[0],scale=hyper_gamma[1])

				def prior_sample(theta):
					# Sample from the prior
					# It will be used to marginalize the
					# source parameters
					result = theta[0] + theta[1]*eff.rvs(gamma=theta[2],size=self.M)
					return result

				def hp_transform(u):
					# Transform unit cube hyper prior
					x = np.zeros_like(u)
					x[0] = hp_loc.ppf(u[0])
					x[1] = hp_scl.ppf(u[1])
					x[2] = hp_gamma.ppf(u[2])
					return x

			else:
				self.D = 2
				self.names = ["loc","scl"]

				def prior_sample(theta):
					# Sample from the prior
					# It will be used to marginalize the
					# source parameters
					result = theta[0] + theta[1]*eff.rvs(gamma=parameters["gamma"],size=self.M)
					return result

				def hp_transform(u):
					# Transform unit cube hyper prior
					x = np.zeros_like(u)
					x[0] = hp_loc.ppf(u[0])
					x[1] = hp_scl.ppf(u[1])
					return x



		elif prior == "King":
			if parameters["rt"] is None:
				self.D = 3
				self.names = ["loc","scl","rt"]

				hp_x = st.gamma(a=2.0,scale=2.0/hyper_gamma[0])

				def prior_sample(theta):
					# Sample from the prior
					# It will be used to marginalize the
					# source parameters
					result = theta[0] + theta[1]*king.rvs(rt=theta[2],size=self.M)
					return result

				def hp_transform(u):
					# Transform unit cube hyper prior
					x = np.zeros_like(u)
					x[0] = hp_loc.ppf(u[0])
					x[1] = hp_scl.ppf(u[1])
					x[2] = 1.0 + hp_x.ppf(u[2])
					return x

			else:
				self.D = 2
				self.names = ["loc","scl"]

				def prior_sample(theta):
					# Sample from the prior
					# It will be used to marginalize the
					# source parameters
					result = theta[0] + theta[1]*king.rvs(rt=parameters["rt"],size=self.M)
					return result

				def hp_transform(u):
					# Transform unit cube hyper prior
					x = np.zeros_like(u)
					x[0] = hp_loc.ppf(u[0])
					x[1] = hp_scl.ppf(u[1])
					return x

		else:
			sys.exit("The specified prior is not supported. Check spelling.")

		self.prior_sample    = prior_sample
		self.hp_transform    = hp_transform

	#=============== Likelihood =========================================
	def logsumdensity(self,datum,x):
		# Log of the marginalized likelihood
		z   = ((x-datum[0])/datum[1])**2
		cte = 2.*np.log(datum[1]) + np.log(2.0*np.pi)
		ld  = -0.5*(z + cte)
		res = logsumexp(ld)
		return res

	def loglike(self,theta):
		"""The log-likelihood function."""
		rvs   = self.prior_sample(theta)
		trvs  = self.Transformation(rvs)
		lmarg = np.apply_along_axis(self.logsumdensity,1,self.data,trvs) - self.logM
		return lmarg.sum()

	def run(self,dlogz,nlive,bound="single",print_progress=False):
		""""Run the nested sampler algorithm"""
		sampler = dynesty.NestedSampler(self.loglike,self.hp_transform,self.D,bound=bound,nlive=nlive)
		sampler.run_nested(dlogz=dlogz,print_progress=print_progress)
		return sampler.results

	def parameters_statistics(self,results):
		# Compute quantiles of the parameters posterior distribution

		# Extract sampling results.
		samples = results.samples
		weights = np.exp(results.logwt - results.logz[-1])  # normalized weights

		quantiles = [dynesty.utils.quantile(samps,self.quantiles, weights=weights)
			for samps in samples.T]

		summary = pn.DataFrame(data=quantiles, index=self.names, columns=["lower","median","upper"])
		return summary

	def plots(self,results,file,figsize=(8,8)):
		pdf = PdfPages(filename=file)

		# Plot a summary of the run.
		plt.figure(figsize=figsize)
		rfig, raxes = dynesty.plotting.runplot(results)
		pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
		plt.close()

		# Plot traces and 1-D marginalized posteriors.
		plt.figure(figsize=figsize)
		tfig, taxes = dynesty.plotting.traceplot(results)
		pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
		plt.close()

		# Plot the 2-D marginalized posteriors.
		plt.figure(figsize=figsize)
		cfig, caxes = dynesty.plotting.cornerplot(results)
		pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
		plt.close()

		pdf.close()
####################################################################################################

class Evidence3D():
	'''
	Infer evidence of a model
	'''
	def __init__(self,mu_data,sg_data,
		prior="Gaussian",
		parameters={"location":None,"scale": None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		N_samples=None,
		M_samples=10000,
		transformation="mas",
		quantiles=[0.05,0.5,0.95],
		seed=12345):
		np.random.seed(seed)

		self.M    = M_samples
		self.logM = np.log(M_samples)

		self.quantiles = quantiles

		if parameters["location"] is not None or parameters["scale"] is not None:
			sys.exit("This modules works only when location and scale are set free")

		#========================= Data ==================================================
		#-------- Assume independence amongst sources ------------------
		data = np.column_stack((mu_data,np.sqrt(np.diag(sg_data))))
		
		#----- Use all data or a random sample of it --------
		if N_samples is None:
			self.N    = len(mu_data)
			self.data = data
		else:
			self.N    = N_samples
			idx       = np.random.choice(np.arange(0,len(mu_data),1),self.N)
			self.data = data[idx]

		if self.N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")
		#================================================================================

		#============= Transformations =============================================
		assert transformation == "pc","3D model only works with 'pc' transformation"
		#===========================================================================

		#================ Hyper-parameters =====================================
		#----------------- Mixture prior families ----------------------------
		if prior in ["GMM","CGMM"]:
			#------------- Shapes -------------------------
			n_gauss = len(hyper_delta)
			#----------------------------------------------

			#----------- Locations ------------------------------------------
			if parameters["location"] is None:
				if prior in ["CGMM"]:
					#----------------- Concentric prior --------------------
					location = [ pm.Normal("loc_{0}".format(j),
								mu=hyper_alpha[j][0],
								sigma=hyper_alpha[j][1]) for j in range(3) ]

					loci = pm.math.stack(location,axis=1)

					for i in range(n_gauss):
						loc  = tt.set_subtensor(loc[i],loci)
					#---------------------------------------------------------

				else:
					#----------- Non-concentric prior ----------------------------
					location = [ pm.Normal("loc_{0}".format(j),
								mu=hyper_alpha[j][0],
								sigma=hyper_alpha[j][1],
								shape=n_gauss) for j in range(3) ]
					
					loc = pm.math.stack(location,axis=1)
					#---------------------------------------------------------
				#-------------------------------------------------------------------
			else:
				for i in range(n_gauss):
					loc  = tt.set_subtensor(loc[i],np.array(parameters["location"][i]))

			#---------- Covariance matrices -----------------------------------
			if parameters["scale"] is None:
				for i in range(n_gauss):
					choli, corri, stdsi = pm.LKJCholeskyCov("scl_{0}".format(i), 
										n=3, eta=hyper_eta, 
										sd_dist=pm.Gamma.dist(
										alpha=2.0,beta=1.0/hyper_beta),
										compute_corr=True)
				
					chol = tt.set_subtensor(chol[i],choli)

			else:
				sys.exit("Not yet implemented.")
			#--------------------------------------------------------------------
		#---------------------------------------------------------------------------------

		#-------------- Non-mixture prior families ----------------------------------
		else:
			# You need transformations (bijectors) from the hyper-cube [0,1]
			# To the parameter space
			#--------- Location ----------------------------------
			hps_loc = [ st.norm(loc=hyper_alpha[i][0],
								scale=hyper_alpha[i][1])
						for i in range(3) ]
			#------------------------------------------------------

			#---------- Covariance matrix ------------------------------------
			chol, corr, stds = pm.LKJCholeskyCov("scl", n=3, eta=hyper_eta, 
					sd_dist=pm.Gamma.dist(alpha=2.0,beta=1.0/hyper_beta),
					compute_corr=True)
			#--------------------------------------------------------------
		#----------------------------------------------------------------------------
		#==============================================================================

		#================ Hyper-prior =====================================
		hp_loc = st.norm(loc=hyper_alpha[0],scale=hyper_alpha[1])
		hp_scl = st.gamma(a=2.0,scale=hyper_beta[0]/2.0)
		#========================================================================

		#================= Prior ========================================================
		#--------- Cluster oriented prior-----------------------------------------------
		if prior == "Uniform":
			self.D = 2
			self.names = ["loc","scl"]

			def prior_sample(theta):
				result = theta[0] + theta[1]*np.random.uniform(-1.0,1.0,size=self.M)
				return result

			def hp_transform(u):
				x = np.zeros_like(u)
				x[0] = hp_loc.ppf(u[0])
				x[1] = hp_scl.ppf(u[1])
				return x

		elif prior == "Gaussian":
			self.D = 2
			self.names = ["loc","scl"]

			def prior_sample(theta):
				result = theta[0] + theta[1]*st.norm.rvs(size=self.M)
				return result

			def hp_transform(u):
				x = np.zeros_like(u)
				x[0] = hp_loc.ppf(u[0])
				x[1] = hp_scl.ppf(u[1])
				return x

		elif prior == "GMM":

			if parameters["weights"] is None:

				# Components in the mixture
				G      = len(hyper_delta)

				self.D     = G*3
				self.names = sum([["weights_{0}".format(i) for i in range(G)],["loc_{0}".format(i) for i in range(G)],["scl_{0}".format(i) for i in range(G)]],[])

				def prior_sample(theta):
					samples = st.norm.rvs(size=(self.M,G))

					frc = theta[:G]
					loc = theta[G:(2*G)]
					scl = theta[(2*G):]

					result = np.dot(loc + scl*samples,frc)
					return result

				def hp_transform(u):
					# Transform unit cube hyper prior
					x = np.zeros_like(u)
					x[:G]      = u[:G]
					x[G:(2*G)] = hp_loc.ppf(u[G:(2*G)])
					x[(2*G):]  = hp_scl.ppf(u[(2*G):])

					x[:G] = x[:G]/np.sum(x[:G]) 
					return x

			else:
				G = len(hyper_delta)

				self.D     = 2*G
				self.names = sum([["loc_{0}".format(i) for i in range(G)],["scl_{0}".format(i) for i in range(G)]],[])

				def prior_sample(theta):
					samples = st.norm.rvs(size=(self.M,G))

					frc = np.array(parameters["weights"])
					loc = theta[:G]
					scl = theta[G:(2*G)]

					result = np.dot(loc + scl*samples,frc)
					return result

				def hp_transform(u):
					# Transform unit cube hyper prior
					x = np.zeros_like(u)
					x[:G]      = hp_loc.ppf(u[:G])
					x[G:(2*G)] = hp_scl.ppf(u[G:(2*G)])
					return x

		elif prior == "EFF":

			if parameters["gamma"] is None:
				self.D = 3
				self.names = ["loc","scl","gamma"]

				if hyper_gamma[0] < 2.0:
					sys.exit("Setting hyper_gamma[0] to values < 2.0 leads to extremely inefficient sampling")

				a, b = (2.0 - hyper_gamma[0]) / hyper_gamma[1], (10. - hyper_gamma[0]) / hyper_gamma[1]

				hp_gamma = st.truncnorm(a=a,b=b,loc=hyper_gamma[0],scale=hyper_gamma[1])

				def prior_sample(theta):
					# Sample from the prior
					# It will be used to marginalize the
					# source parameters
					result = theta[0] + theta[1]*eff.rvs(gamma=theta[2],size=self.M)
					return result

				def hp_transform(u):
					# Transform unit cube hyper prior
					x = np.zeros_like(u)
					x[0] = hp_loc.ppf(u[0])
					x[1] = hp_scl.ppf(u[1])
					x[2] = hp_gamma.ppf(u[2])
					return x

			else:
				self.D = 2
				self.names = ["loc","scl"]

				def prior_sample(theta):
					# Sample from the prior
					# It will be used to marginalize the
					# source parameters
					result = theta[0] + theta[1]*eff.rvs(gamma=parameters["gamma"],size=self.M)
					return result

				def hp_transform(u):
					# Transform unit cube hyper prior
					x = np.zeros_like(u)
					x[0] = hp_loc.ppf(u[0])
					x[1] = hp_scl.ppf(u[1])
					return x



		elif prior == "King":
			if parameters["rt"] is None:
				self.D = 3
				self.names = ["loc","scl","rt"]

				hp_x = st.gamma(a=2.0,scale=2.0/hyper_gamma[0])

				def prior_sample(theta):
					# Sample from the prior
					# It will be used to marginalize the
					# source parameters
					result = theta[0] + theta[1]*king.rvs(rt=theta[2],size=self.M)
					return result

				def hp_transform(u):
					# Transform unit cube hyper prior
					x = np.zeros_like(u)
					x[0] = hp_loc.ppf(u[0])
					x[1] = hp_scl.ppf(u[1])
					x[2] = 1.0 + hp_x.ppf(u[2])
					return x

			else:
				self.D = 2
				self.names = ["loc","scl"]

				def prior_sample(theta):
					# Sample from the prior
					# It will be used to marginalize the
					# source parameters
					result = theta[0] + theta[1]*king.rvs(rt=parameters["rt"],size=self.M)
					return result

				def hp_transform(u):
					# Transform unit cube hyper prior
					x = np.zeros_like(u)
					x[0] = hp_loc.ppf(u[0])
					x[1] = hp_scl.ppf(u[1])
					return x

		else:
			sys.exit("The specified prior is not supported. Check spelling.")

		self.prior_sample    = prior_sample
		self.hp_transform    = hp_transform

	#=============== Likelihood =========================================
	def logsumdensity(self,datum,x):
		# Log of the marginalized likelihood
		z   = ((x-datum[0])/datum[1])**2
		cte = 2.*np.log(datum[1]) + np.log(2.0*np.pi)
		ld  = -0.5*(z + cte)
		res = logsumexp(ld)
		return res

	def loglike(self,theta):
		"""The log-likelihood function."""
		rvs   = self.prior_sample(theta)
		trvs  = self.Transformation(rvs)
		lmarg = np.apply_along_axis(self.logsumdensity,1,self.data,trvs) - self.logM
		return lmarg.sum()

	def run(self,dlogz,nlive,bound="single",print_progress=False):
		""""Run the nested sampler algorithm"""
		sampler = dynesty.NestedSampler(self.loglike,self.hp_transform,self.D,bound=bound,nlive=nlive)
		sampler.run_nested(dlogz=dlogz,print_progress=print_progress)
		return sampler.results

	def parameters_statistics(self,results):
		# Compute quantiles of the parameters posterior distribution

		# Extract sampling results.
		samples = results.samples
		weights = np.exp(results.logwt - results.logz[-1])  # normalized weights

		quantiles = [dynesty.utils.quantile(samps,self.quantiles, weights=weights)
			for samps in samples.T]

		summary = pn.DataFrame(data=quantiles, index=self.names, columns=["lower","median","upper"])
		return summary

	def plots(self,results,file,figsize=(8,8)):
		pdf = PdfPages(filename=file)

		# Plot a summary of the run.
		plt.figure(figsize=figsize)
		rfig, raxes = dynesty.plotting.runplot(results)
		pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
		plt.close()

		# Plot traces and 1-D marginalized posteriors.
		plt.figure(figsize=figsize)
		tfig, taxes = dynesty.plotting.traceplot(results)
		pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
		plt.close()

		# Plot the 2-D marginalized posteriors.
		plt.figure(figsize=figsize)
		cfig, caxes = dynesty.plotting.cornerplot(results)
		pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
		plt.close()

		pdf.close()