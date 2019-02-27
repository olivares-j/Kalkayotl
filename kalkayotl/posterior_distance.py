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
import numpy as np
import scipy.stats as st

class Posterior(object):
	"""
	This class provides the posterior distribution of the distance
	Arguments:
	string 'prior':       Type of prior
	float  'prior_loc':   Location of the prior distribution in pc
						  Assumed zero for Half-Gaussian, Half-Cauchy and EDSD
	float  'prior_scale': Scale of the prior distribution in pc
	"""

	def __init__(self,prior="Uniform",prior_loc=0,prior_scale=100,zero_point=0.0):

		self.prior_loc   = prior_loc
		self.prior_scl   = prior_scale
		self.zero_point  = zero_point


		######################### PRIORS #######################################
		#================ Cluster specific priors ==============================
		if prior == "Uniform" :
			self.lnprior = self.Uniform

			self.pos0 = [st.uniform.rvs(loc=self.prior_loc,scale=self.prior_scl,
						size=(1)) for i in range(self.nwalkers)]

		if prior == "Gaussian" :
			self.lnprior = self.Gaussian		

			self.pos0 = [st.norm.rvs(loc=self.prior_loc,scale=self.prior_scl,
						size=(1)) for i in range(self.nwalkers)]
		
		if prior == "Cauchy" :
			self.lnprior = self.Cauchy
			self.pos0 = [st.cauchy.rvs(loc=self.prior_loc,scale=self.prior_scl,
						size=(1)) for i in range(self.nwalkers)]

		if prior == "Half-Gaussian" :
			self.lnprior = self.HalfGaussian		

			self.pos0 = [st.halfnorm.rvs(loc=0.0,scale=self.prior_scl,
						size=(1)) for i in range(self.nwalkers)]
		
		if prior == "Half-Cauchy" :
			self.lnprior = self.HalfCauchy

			self.pos0 = [st.halfcauchy.rvs(loc=0.0,scale=self.prior_scl,
						size=(1)) for i in range(self.nwalkers)]

		if prior == "EDSD" :
			self.lnprior = self.EDSD

			self.pos0 = [st.uniform.rvs(loc=0.0,scale=self.prior_scl,
						size=(1)) for i in range(self.nwalkers)]


	def Uniform(self,theta):
		""" 
		Uniform prior
		"""
		if theta > self.prior_scl or theta < self.prior_loc:
			return -np.inf
		else:
			return st.uniform.logpdf(theta,loc=self.prior_loc,scale=self.prior_scl)

	def Gaussian(self,theta):
		"""
		Gaussian prior
		"""
		return st.norm.logpdf(theta,loc=self.prior_loc,scale=self.prior_scl)

	def Cauchy(self,theta):
		"""
		Cauchy prior
		"""
		return st.cauchy.logpdf(theta,loc=self.prior_loc,scale=self.prior_scl)

	def HalfGaussian(self,theta):
		"""
		Half Gaussian prior
		"""
		return st.halfnorm.logpdf(theta,loc=0.0,scale=self.prior_scl)

	def HalfCauchy(self,theta):
		"""
		Half Cauchy prior
		"""
		return st.halfcauchy.logpdf(theta,loc=0.0,scale=self.prior_scl)

	def EDSD(self,theta):
		"""
		Exponentialy decreasing space density prior
		Bailer-Jones 2015
		"""

		log_prior = -1.0*(theta/self.prior_scl) - np.log((2.0*(self.prior_scl**3))*(theta**2)) 
		return log_prior

	################ POSTERIOR#######################
	def __call__(self,theta, pllx, u_pllx):
		if theta <= 0.0:
			return -np.inf
		else:
			corrected_pllx = pllx + self.zero_point
			return self.lnprior(theta) + st.norm.logpdf(corrected_pllx,loc=1.0/theta,scale=u_pllx)
