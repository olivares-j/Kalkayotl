'''
Copyright 2018 Javier Olivares Romero

This file is part of Kalkayotl.

    Kalkayotl is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyAspidistra is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyAspidistra.  If not, see <http://www.gnu.org/licenses/>.
'''
from __future__ import absolute_import, unicode_literals, print_function
import emcee
import numpy as np
import scipy.stats as st

class parallax2distance:
	"""
	This class provides flexibility to infer the distance distribution given the parallax and its uncertainty
	"""
	def __init__(self,N_iter=1000,nwalkers=10,prior="Uniform",prior_loc=0,prior_scale=100,burnin_frac=0.2,quantiles=[2.5,97.5]):

		self.N_iter      = N_iter
		self.nwalkers    = nwalkers
		self.prior_loc   = prior_loc
		self.prior_scale = prior_scale
		self.burnin_frac = burnin_frac
		self.quantiles   = quantiles
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
		Median = np.median(sample.flatten())
		#----- SD ------
		SD  = np.std(sample.flatten())
		#---- CI 95%
		CI  = np.percentile(sample.flatten(),q=self.quantiles)
		#------ autocorrelation time
		int_time = emcee.autocorr.integrated_time(sample.flatten(),tol=0)
		return MAP,Median,SD,CI,int_time,sample
