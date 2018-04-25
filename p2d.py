
# import myemcee as emcee
import emcee
import numpy as np
import scipy.stats as st

class parallax2distance:
	"""
	This class provides flexibility to infer the distance duistribution gien the parallax and its uncertainty
	"""
	def __init__(self,N_iter=1000,nwalkers=10,prior="Uniform",prior_loc=0,prior_scale=100,burnin_frac=0.2):

		self.N_iter      = N_iter
		self.nwalkers    = nwalkers
		self.prior_loc   = prior_loc
		self.prior_scale = prior_scale
		self.burnin_frac = burnin_frac
		########### PRIORS #####################
		if prior=="Uniform" :
			def lnprior(theta):
				if theta > prior_scale :
					return -np.inf
				else:
					return st.uniform.logpdf(theta,loc=prior_loc,scale=prior_scale)

			self.pos0 = [st.uniform.rvs(loc=prior_loc,scale=prior_scale,size=(1)) for i in range(self.nwalkers)]

		if prior=="Gaussian" :		
			def lnprior(theta):
				"""
				Truncated Normal distribution
				"""
				a = (0.0 - prior_loc) / prior_scale
				return st.truncnorm.logpdf(theta,a=a,b=np.inf,loc=prior_loc,scale=prior_scale)

			a = (0.0 - prior_loc) / prior_scale
			self.pos0 = [st.truncnorm.rvs(a=a,b=np.inf,loc=prior_loc,scale=prior_scale,size=(1)) for i in range(self.nwalkers)]
		
		if prior=="Cauchy" :
			def lnprior(theta):
				"""
				Truncated Cauchy distribution
				"""
				cte=1.0/(1.0-st.cauchy.cdf(0.0,loc=prior_loc,scale=prior_scale))
				return cte*st.cauchy.logpdf(theta,loc=prior_loc,scale=prior_scale)

			self.pos0 = [np.abs(st.cauchy.rvs(loc=prior_loc,scale=prior_scale,size=(1))) for i in range(self.nwalkers)]

		if prior=="EDSD" :
			def lnprior(theta):
				# Exponentialy decreasing space density prior
				# Bailer-Jones 2015
				L = prior_scale # Scale distance in parsecs
				pri = (1.0/(2.0*(L**3)))*(theta**2)*np.exp(-(theta/L))
				return np.log(pri)

			self.pos0 = [st.uniform.rvs(loc=prior_loc,scale=prior_scale,size=(1)) for i in range(self.nwalkers)]

		self.lnprior = lnprior

	################ POSTERIOR#######################
	def lnprob(self,theta, pllx, u_pllx):
		if theta <= 0.0:
			return -np.inf
		else:
			return self.lnprior(theta) + st.norm.logpdf(pllx,loc=1.0/theta,scale=u_pllx)

	#################### RUN THE SAMPLER ####################################

	def run(self,plx,uplx):
		sampler = emcee.EnsembleSampler(self.nwalkers,1, self.lnprob, args=[plx, uplx])
		sampler.run_mcmc(self.pos0, self.N_iter)

		sample = sampler.chain[:,int(self.burnin_frac*self.N_iter):,0]

		#-----MAP ----
		mins,maxs = np.min(sample),np.max(sample)
		x = np.linspace(mins,maxs,num=100)
		gkde = st.gaussian_kde(sample.flatten())
		MAP = x[np.argmax(gkde(x))]
		#-----------
		Mean = np.mean(sample.flatten())
		#----- SD ------
		SD  = np.std(sample.flatten())
		#---- CI 95%
		CI  = np.percentile(sample.flatten(),q=[2.5,97.5])
		#------ autocorrelation time
		int_time = emcee.autocorr.integrated_time(sample.flatten(),axis=0)
		return MAP,Mean,SD,CI,int_time,sample
