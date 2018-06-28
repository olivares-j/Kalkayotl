'''
Copyright 2018 Javier Olivares Romero

This file is part of Kalkayotl3D.

    Kalkayotl3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyAspidistra is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kalkayotl3D.  If not, see <http://www.gnu.org/licenses/>.
'''
from __future__ import absolute_import,division,print_function
import sys
# import myemcee as emcee
import emcee
import numpy as np
import scipy.stats as st

class posterior_adw:
	"""
	This class provides flexibility to infer the posterior distribution 
	"""
	def __init__(self,datum,nwalkers=30,prior="Uniform",prior_loc=0,prior_scale=100,burnin_frac=0.2,
		quantiles=[2.5,97.5]):

		self.nwalkers    = nwalkers
		self.prior_loc   = prior_loc
		self.prior_scale = prior_scale
		self.burnin_frac = burnin_frac
		self.quantiles   = quantiles
		self.ndim        = 3


		log_prior_ad     = (1.0/360)*(1.0/180)
		########### PRIORS #####################
		if prior=="Uniform" :
			def lnprior(theta):
				if theta[2] > prior_scale :
					return -np.inf
				else:
					return st.uniform.logpdf(theta[2],loc=prior_loc,scale=prior_scale)+log_prior_ad

			self.pos0 = [np.array([st.uniform.rvs(loc=0,scale=360,size=1)[0],
								   st.uniform.rvs(loc=-90,scale=180,size=1)[0],
								   st.uniform.rvs(loc=prior_loc,scale=prior_scale,size=1)[0]]) for i in range(self.nwalkers)]


		if prior=="Gaussian" :		
			def lnprior(theta):
				"""
				Truncated Normal distribution
				"""
				a = (0.0 - prior_loc) / prior_scale
				return st.truncnorm.logpdf(theta[2],a=a,b=np.inf,loc=prior_loc,scale=prior_scale)+log_prior_ad

			a = (0.0 - prior_loc) / prior_scale

			self.pos0 = [np.array([st.uniform.rvs(loc=0,scale=360,size=1)[0],
								   st.uniform.rvs(loc=-90,scale=180,size=1)[0],
								   st.truncnorm.rvs(a=a,b=np.inf,loc=prior_loc,scale=prior_scale,size=(1))[0]]) for i in range(self.nwalkers)]
		
		if prior=="Cauchy" :
			def lnprior(theta):
				"""
				Truncated Cauchy distribution
				"""
				cte=1.0/(1.0-st.cauchy.cdf(0.0,loc=prior_loc,scale=prior_scale))
				return cte*st.cauchy.logpdf(theta[2],loc=prior_loc,scale=prior_scale)+log_prior_ad

			self.pos0 = [np.array([st.uniform.rvs(loc=0,scale=360,size=1)[0],
								   st.uniform.rvs(loc=-90,scale=180,size=1)[0],
								   np.abs(st.cauchy.rvs(loc=prior_loc,scale=prior_scale,size=(1)))[0]]) for i in range(self.nwalkers)]

		if prior=="EDSD" :
			def lnprior(theta):
				# Exponentialy decreasing space density prior
				# Bailer-Jones 2015
				L = prior_scale # Scale distance in parsecs
				pri = (1.0/(2.0*(L**3)))*(theta[2]**2)*np.exp(-(theta[2]/L))
				return np.log(pri)+log_prior_ad

			self.pos0 = [np.array([st.uniform.rvs(loc=0,scale=360,size=1)[0],
								   st.uniform.rvs(loc=-90,scale=180,size=1)[0],
								   st.uniform.rvs(loc=prior_loc,scale=prior_scale,size=(1))[0]]) for i in range(self.nwalkers)]

		self.lnprior = lnprior

		#-----------------------------------------------------------
		def lmn(mu):
			ra,dec,pax,u_ra,u_dec,u_pax,corr_ra_dec,corr_ra_pax,corr_dec_pax = datum
			corr      = np.zeros((self.ndim,self.ndim))
			corr[0,1] = corr_ra_dec
			corr[0,2] = corr_ra_pax
			corr[1,2] = corr_dec_pax

			corr      = corr + corr.T + np.eye(self.ndim)

			cov       = np.diag([u_ra,u_dec,u_pax]).dot(corr.dot(np.diag([u_ra,u_dec,u_pax])))

			observed  = np.array([ra,dec,pax])

			x = observed - mu

			try:
				inv = np.linalg.inv(cov)
			except Exception as e:
				sys.exit(e)
			else:
				pass
			finally:
				pass

			try:
				s,logdet = np.linalg.slogdet(cov)
			except Exception as e:
				sys.exit(e)
			else:
				pass
			finally:
				pass

			if s <= 0:
				sys.exit("Negative determinant!")

			arg     = -0.5*np.dot(x.T,inv.dot(x))
			log_den = -0.5*(3.0*np.log(2.0*np.pi) + logdet)

			return arg + log_den

		self.logpdf_multivariate_normal = lmn

	################ POSTERIOR#######################
	def lnprob(self,theta):
		if theta[2] <= 0.0:
			return -np.inf
		else:
			true      = np.array([theta[0],theta[1],1.0/theta[2]])
			return self.lnprior(theta) + self.logpdf_multivariate_normal(true)

	#################### RUN THE SAMPLER ####################################

	def run(self,N_iter):
		sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim, self.lnprob)
		sampler.run_mcmc(self.pos0,N_iter)
		# print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

		sample = sampler.chain[:,int(self.burnin_frac*N_iter):,:]

		#-----MAP ----
		ind_map = np.unravel_index(np.argmax(sampler.lnprobability),np.shape(sampler.lnprobability))
		MAP  = sampler.chain[ind_map[0],ind_map[1],:]
		#-----------
		Median = np.median(sample,axis=(0,1))
		#----- SD ------
		SD  = np.std(sample,axis=(0,1))
		#---- CI 95%
		CI  = np.percentile(sample,axis=(0,1),q=self.quantiles)
		#------ autocorrelation time
		int_time = emcee.autocorr.integrated_time(sample[:,:,2].flatten(),axis=0,c=5)
		return MAP,Median,SD,CI,int_time,sample,np.mean(sampler.acceptance_fraction)
