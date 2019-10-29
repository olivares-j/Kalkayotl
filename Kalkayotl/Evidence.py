import sys
import dynesty
import numpy as np
import pandas as pn
import scipy.stats as st
from scipy.special import logsumexp,hyp2f1,gamma as gamma_function

from dynesty import plotting as dyplot

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Kalkayotl.Transformations import Iden,pc2mas
from Kalkayotl.EFF import eff
from Kalkayotl.King import king


################################## Evidence 1D ####################################
class Evidence1D():
	'''
	Infer evidence of a model
	'''
	def __init__(self,mu_data,sg_data,
		prior="Gaussian",
		hyper_alpha=[[0,10]],
		hyper_beta=[0.5],
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

		#============= Transformations ====================================

		if transformation is "mas":
			self.Transformation = Iden

		elif transformation is "pc":
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
		if prior is "Uniform":
			self.D = 2
			self.names = ["loc_0","scl_0"]

			def prior_sample(parameters):
				# Sample from the prior
				# It will be used to marginalize the
				# source parameters
				result =  np.abs(st.uniform.rvs(
					loc=parameters[0]-parameters[1],
					scale=2*parameters[1],
					size=self.M))
				return result

			def hp_transform(u):
				# Transform unit cube hyper prior
				x = np.zeros_like(u)
				x[0] = hp_loc.ppf(u[0])
				x[1] = hp_scl.ppf(u[1])
				return x
			

		elif prior is "Cauchy":
			self.D = 2
			self.names = ["loc_0","scl_0"]

			def prior_sample(parameters):
				# Sample from the prior
				# It will be used to marginalize the
				# source parameters
				result =  np.abs(st.cauchy.rvs(
					loc=parameters[0],
					scale=parameters[1],
					size=self.M))
				return result

			def hp_transform(u):
				# Transform unit cube hyper prior
				x = np.zeros_like(u)
				x[0] = hp_loc.ppf(u[0])
				x[1] = hp_scl.ppf(u[1])
				return x

		elif prior is "Gaussian":
			self.D = 2
			self.names = ["loc_0","scl_0"]

			def prior_sample(parameters):
				# Sample from the prior
				# It will be used to marginalize the
				# source parameters
				result =  np.abs(st.norm.rvs(
					loc=parameters[0],
					scale=parameters[1],
					size=self.M))
				return result

			def hp_transform(u):
				# Transform unit cube hyper prior
				x = np.zeros_like(u)
				x[0] = hp_loc.ppf(u[0])
				x[1] = hp_scl.ppf(u[1])
				return x

		elif prior is "GMM":
			# Components in the mixture
			G      = len(hyper_delta)

			self.D     = G*3 -1
			self.names = sum([["weights_{0}".format(i) for i in range(G-1)],["loc_{0}".format(i) for i in range(G)],["scl_{0}".format(i) for i in range(G)]],[])

			def prior_sample(parameters):
				# Sample from the prior
				# It will be used to marginalize the
				# source parameters
				szs = np.zeros(G,dtype='int')
				for i in range(G-1):
					szs[i] = int(np.floor(parameters[i]*self.M))
				szs[-1] = int(self.M - sum(szs))

				loc = parameters[(G-1):(2*G -1)]
				scl = parameters[(2*G -1):]

				result = []
				for j in range(G):
					result.append(np.abs(st.norm.rvs(
						loc=loc[j],scale=scl[j],
						size=szs[j])))

				res = np.concatenate(result)
				return res

			def hp_transform(u):
				# Transform unit cube hyper prior
				x = np.zeros_like(u)
				x[:(G-1)]         = u[:(G-1)]
				x[(G-1):(2*G -1)] = hp_loc.ppf(u[(G-1):(2*G -1)])
				x[(2*G -1):]      = hp_scl.ppf(u[(2*G -1):])
				return x

		elif prior is "EFF":
			self.D = 3
			self.names = ["loc_0","scl_0","gamma"]

			if hyper_gamma[0] < 2.0:
				sys.exit("Setting hyper_gamma[0] to values < 2.0 leads to extremely inefficient sampling")

			a, b = (2.0 - hyper_gamma[0]) / hyper_gamma[1], (100. - hyper_gamma[0]) / hyper_gamma[1]

			hp_gamma = st.truncnorm(a=a,b=b,loc=hyper_gamma[0],scale=hyper_gamma[1])

			def prior_sample(parameters):
				# Sample from the prior
				# It will be used to marginalize the
				# source parameters
				result =  np.abs(eff.rvs(
					r0=parameters[0],
					rc=parameters[1],
					gamma=parameters[2],
					size=self.M))
				return result

			def hp_transform(u):
				# Transform unit cube hyper prior
				x = np.zeros_like(u)
				x[0] = hp_loc.ppf(u[0])
				x[1] = hp_scl.ppf(u[1])
				x[2] = hp_gamma.ppf(u[2])
				return x


		elif prior is "King":
			self.D = 3
			self.names = ["loc_0","scl_0","rt"]

			hp_rt = st.halfnorm(loc=0.0,scale=hyper_gamma[0])

			def prior_sample(parameters):
				# Sample from the prior
				# It will be used to marginalize the
				# source parameters
				result =  np.abs(king.rvs(
					r0=parameters[0],
					rc=parameters[1],
					rt=parameters[2],
					size=self.M))
				return result

			def hp_transform(u):
				# Transform unit cube hyper prior

				x = np.zeros_like(u)
				x[0] = hp_loc.ppf(u[0])
				x[1] = hp_scl.ppf(u[1])
				x[2] = x[1] + hp_rt.ppf(u[2])
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

	def loglike(self,parameters):
		"""The log-likelihood function."""
		rvs   = self.prior_sample(parameters)
		trvs  = self.Transformation(rvs)
		lmarg = np.apply_along_axis(self.logsumdensity,1,self.data,trvs) - self.logM
		return lmarg.sum()

	def run(self,dlogz,nlive,bound="single"):
		""""Run the nested sampler algorithm"""
		sampler = dynesty.NestedSampler(self.loglike,self.hp_transform,self.D,bound=bound,nlive=nlive)
		sampler.run_nested(dlogz=dlogz)
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