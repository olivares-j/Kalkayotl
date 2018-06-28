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
    along with Kalkayotl.  If not, see <http://www.gnu.org/licenses/>.
'''
from __future__ import absolute_import, unicode_literals, print_function
import sys
import emcee
import corner
import numpy as np
import scipy.stats as st
import scipy.integrate as integrate

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class pps:
	"""
	This class provides flexibility to infer the parameters of the prior distribution for distance estimation
	"""
	def __init__(self,parallax,parallax_uncertainty,nwalkers=10,distance_prior="Uniform",burnin_frac=0.2,quantiles=[0.16, 0.5, 0.84]):

		self.nwalkers    = nwalkers
		self.pax         = np.column_stack((parallax,parallax_uncertainty))
		self.burnin_frac = burnin_frac
		self.quantiles   = np.atleast_1d(quantiles)
		self.names       = np.array(["Location [pc]","Scale [pc]"])


		#----------- Hyper parameters ---------
		hyp_loc_min      = 0
		hyp_loc_max      = 1e4
		hyp_scl_min      = 1e-2
		hyp_scl_max      = 1e2
		########### PRIORS of DISTANCE #####################
		if distance_prior=="Uniform" :
			self.ndim = 2
			# theta[0] = location
			# theta[1] = scale

			def llik_star(theta_pax):

				#--------- Likelihood --------------------------------------------------------
				# The following likelihood has been marginalised over the distance.
				# It is the integral from zero to infinity of the product of the likelihood times the distance prior.
				loc   = theta_pax[0]
				scl   = theta_pax[1]
				pax   = theta_pax[2]
				v_pax = theta_pax[3]**2
				
				integral = integrate.quad(lambda x:np.exp(-0.5*((pax-(1.0/x))**2)/(2.0*v_pax)),loc,loc+scl)
				marginal = (1.0/np.sqrt(2.0*np.pi*v_pax))*(1.0/scl)*integral[0]

				return np.log(marginal)

			def lnprob(theta):

				#------- Support --------------------
				if (theta[0] < hyp_loc_min or
				   theta[0] > hyp_loc_max or 
				   theta[1] < hyp_scl_min ) :
					return -np.inf
				#------- Prior --------------------------------------------------------------

				lp_theta_0 = st.uniform.logpdf(theta[0],loc=hyp_loc_min,scale=hyp_loc_max)
				lp_theta_1 = st.halfcauchy.logpdf(theta[1],loc=hyp_scl_min,scale=hyp_scl_max)
				
				log_prior  =  lp_theta_0 + lp_theta_1

				thetas  = np.broadcast_to(theta,(self.pax.shape[0],2))

				theta_pax  = np.block([thetas,self.pax])

				log_lik = np.sum(map(llik_star,theta_pax))

				return log_prior + log_lik
					

			self.pos0 = [np.array([st.uniform.rvs(loc=hyp_loc_min,scale=hyp_loc_max,size=1)[0],
								   st.halfcauchy.rvs(loc=hyp_scl_min,scale=hyp_scl_max,size=1)[0]]) for i in range(self.nwalkers)]

			self.lnprob = lnprob

		if distance_prior=="Gaussian" :
			self.ndim = 2
			# theta[0] = location
			# theta[1] = scale

			def llik_star(theta_pax):

				#--------- Likelihood --------------------------------------------------------
				# The following likelihood has been marginalised over the distance.
				# It is the integral from zero to infinity of the product of the likelihood times the distance prior.
				loc   = theta_pax[0]
				vscl  = theta_pax[1]**2
				pax   = theta_pax[2]
				v_pax = theta_pax[3]**2
				
				integral = integrate.quad(lambda x:np.exp(-0.5*((((pax-(1.0/x))**2)/(2.0*v_pax))+(((x-loc)**2)/(2.0*vscl)))),0,hyp_loc_max)
				marginal = (1.0/np.sqrt(2.0*np.pi*v_pax))*(1.0/np.sqrt(2.0*np.pi*vscl))*integral[0]

				return np.log(marginal)

			def lnprob(theta):

				#------- Support --------------------
				if (theta[0] < hyp_loc_min or
				    theta[0] > hyp_loc_max or 
				    theta[1] < hyp_scl_min ) :
					return -np.inf
				#------- Prior --------------------------------------------------------------

				lp_theta_0 = st.uniform.logpdf(theta[0],loc=hyp_loc_min,scale=hyp_loc_max)
				lp_theta_1 = st.halfcauchy.logpdf(theta[1],loc=hyp_scl_min,scale=hyp_scl_max)
				
				log_prior  =  lp_theta_0 + lp_theta_1

				thetas  = np.broadcast_to(theta,(self.pax.shape[0],2))

				theta_pax  = np.block([thetas,self.pax])

				log_lik = np.sum(map(llik_star,theta_pax))

				return log_prior + log_lik
					

			self.pos0 = [np.array([st.uniform.rvs(loc=hyp_loc_min,scale=hyp_loc_max,size=1)[0],
								   st.halfcauchy.rvs(loc=hyp_scl_min,scale=hyp_scl_max,size=1)[0]]) for i in range(self.nwalkers)]

			self.lnprob = lnprob


		if distance_prior=="Cauchy" :
			self.ndim = 2
			# theta[0] = location
			# theta[1] = scale

			def llik_star(theta_pax):

				#--------- Likelihood --------------------------------------------------------
				# The following likelihood has been marginalised over the distance.
				# It is the integral from zero to infinity of the product of the likelihood times the distance prior.
				loc   = theta_pax[0]
				scl   = theta_pax[1]
				pax   = theta_pax[2]
				v_pax = theta_pax[3]**2
				
				integral = integrate.quad(lambda x:np.exp(-0.5*((pax-(1.0/x))**2)/(2.0*v_pax))*(1.0/((x-loc)**2 + scl**2)),0,hyp_loc_max)
				marginal = (1.0/np.sqrt(2.0*np.pi*v_pax))*(scl/np.pi)*integral[0]

				return np.log(marginal)

			def lnprob(theta):

				#------- Support --------------------
				if (theta[0] < hyp_loc_min or
				   theta[0] > hyp_loc_max or 
				   theta[1] < hyp_scl_min ) :
					return -np.inf
				#------- Prior --------------------------------------------------------------

				lp_theta_0 = st.uniform.logpdf(theta[0],loc=hyp_loc_min,scale=hyp_loc_max)
				lp_theta_1 = st.halfcauchy.logpdf(theta[1],loc=hyp_scl_min,scale=hyp_scl_max)
				
				log_prior  =  lp_theta_0 + lp_theta_1

				thetas  = np.broadcast_to(theta,(self.pax.shape[0],2))

				theta_pax  = np.block([thetas,self.pax])

				log_lik = np.sum(map(llik_star,theta_pax))

				return log_prior + log_lik
					

			self.pos0 = [np.array([st.uniform.rvs(loc=hyp_loc_min,scale=hyp_loc_max,size=1)[0],
								   st.halfcauchy.rvs(loc=hyp_scl_min,scale=hyp_scl_max,size=1)[0]]) for i in range(self.nwalkers)]

			self.lnprob = lnprob

		if distance_prior=="EDSD" :
			self.ndim = 1
			self.names = self.names[1]
			# theta[0] = scale

			def llik_star(theta_pax):

				#--------- Likelihood --------------------------------------------------------
				# The following likelihood has been marginalised over the distance.
				# It is the integral from zero to infinity of the product of the likelihood times the distance prior.
				L     = theta_pax[0]
				pax   = theta_pax[2]
				v_pax = theta_pax[3]**2
				
				integral = integrate.quad(lambda x:(x**2)*np.exp(-0.5*(((pax-(1.0/x))**2)/(2.0*v_pax))+ (x/L)),0,hyp_loc_max)
				marginal = (1.0/np.sqrt(2.0*np.pi*v_pax))*(1.0/(2.0*L**3))*integral[0]

				return np.log(marginal)

			def lnprob(theta):

				#------- Support --------------------
				if (theta[0] < hyp_loc_min or
				   theta[0] > hyp_loc_max ) :
					return -np.inf
				#------- Prior --------------------------------------------------------------

				lp_theta_0 = st.uniform.logpdf(theta[0],loc=hyp_loc_min,scale=hyp_loc_max)
		
				log_prior  =  lp_theta_0 

				thetas  = np.repreat(theta,self.pax.shape[0])

				theta_pax  = np.block([thetas,self.pax])

				log_lik = np.sum(map(llik_star,theta_pax))

				return log_prior + log_lik
					

			self.pos0 = [np.array([st.uniform.rvs(loc=hyp_loc_min,scale=hyp_loc_max,size=1)[0]]) for i in range(self.nwalkers)]

			self.lnprob = lnprob


	#################### RUN THE SAMPLER ####################################

	def run(self,N_iter):
		sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim, self.lnprob)
		sampler.run_mcmc(self.pos0, N_iter)

		sample = sampler.chain[:,int(self.burnin_frac*N_iter):,:]
		return sample

	def analyse_sample(self,sample,file_graphs="./analyse_sample.pdf"):
		flat_sample = sample.reshape((sample.shape[0]*sample.shape[1],sample.shape[2]))
		#------ autocorrelation time -----------------------------------------------------
		int_time = emcee.autocorr.integrated_time(flat_sample,axis=0,c=1)
		#--------------------------------------------------------------------------------

		#-------------- Make graphs of the chain -------------------------------------
		pdf = PdfPages(filename=file_graphs)
		f, axarr = plt.subplots(self.ndim, sharex=True)
		axarr[0].set_title('Markov chain')
		axarr[self.ndim-1].set_xlabel("Iteration")
		for i in range(self.ndim):
			axarr[i].plot(sample[:,:,i].T, '-', color='k', alpha=0.3,linewidth=0.3)
			axarr[i].set_ylabel(self.names[i])
			axarr[i].annotate("Autocorrelation time: {0:4.2f}".format(int_time[i]),
            xy=(.65, 0.98), xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=8)

		pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
		plt.close()

		#------------------ Corner plot -------------------------------------------
		# Copyright 2013-2016 Dan Foreman-Mackey & contributors

		figure = corner.corner(flat_sample, labels=self.names,
                       quantiles=self.quantiles,
                       show_titles=True, title_kwargs={"fontsize": 12})
		pdf.savefig(bbox_inches='tight')
		pdf.close()

		quant = np.percentile(flat_sample,list(100.0 * self.quantiles),axis=0)

		return quant
