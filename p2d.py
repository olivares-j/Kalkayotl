
# import myemcee as emcee
import emcee
import numpy as np
import scipy.stats as st
# import random


def p2d(plx,u_plx,N_iter=1000,nwalkers=10,prior="Uniform",mu_prior=100,sigma_prior=10,sg_prior_scale=1,burnin_frac=0.2):
	########### PRIORS #####################
	if prior=="Uniform" :
		def lnprior(theta):
			return st.uniform.logpdf(theta,loc=0,scale=sg_prior_scale*100) #Uniform between 0 and 500 pc

	if prior=="Gaussian" :		
		def lnprior(theta):
			return st.norm.logpdf(theta,loc=mu_prior,scale=sigma_prior*sg_prior_scale)
	
	if prior=="Cauchy" :
		def lnprior(theta):
			return st.cauchy.logpdf(theta,loc=mu_prior,scale=sigma_prior*sg_prior_scale)

	if prior=="EDBJ2015" :
		def lnprior(theta):
			# Exponentialy decreasing space density prior
			# Bailer-Jones 2015
			L = sg_prior_scale*100.0 # Scale distance in parsecs
			pri = (1.0/(2.0*(L**3)))*(theta**2)*np.exp(-(theta/L))
			return np.log(pri)

	################ LIKELIHOOD and POSTERIOR#######################

	def lnlike(theta, pllx, u_pllx):
		return st.norm.logpdf(pllx,loc=1.0/theta,scale=u_pllx)

	def lnprob(theta, pllx, u_pllx):
		return lnprior(theta) + lnlike(theta,pllx,u_pllx)

	########################################################

	pos0 = [np.random.normal(mu_prior,sigma_prior,size=(1)) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers,1, lnprob, args=[plx, u_plx])
	sampler.run_mcmc(pos0, N_iter)

	sample = sampler.chain[:,int(burnin_frac*N_iter):,0]

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
	int_time = emcee.autocorr.integrated_time(sample.flatten(),axis=0,c=2)
	return MAP,Mean,SD,CI,int_time,sample
