'''
Copyright 2019 Javier Olivares Romero

This file is part of Kalkayotl.

    Kalkayotl3D is free software: you can redistribute it and/or modify
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
from __future__ import absolute_import,division,print_function
import sys

import numpy as np
import scipy.stats as st
from posterior_1d import Posterior as posterior_dist

class Posterior:
	"""
	This class provides flexibility to infer the posterior distribution of the 3D
	"""
	def __init__(self,prior=["Uniform","Uniform","Uniform"],
		prior_loc=[180,0,250],
		prior_scale=[180,90,250],
		zero_point=[0.0,0.0,0.0]):

		self.ndim        = 3
		self.prior_loc   = prior_loc
		self.prior_scl   = prior_scale
		self.zero_point  = zero_point

		#================= 1D Prior ===========================

		prior_1d = posterior_dist(prior=prior[2],
								prior_loc=prior_loc[2],
								prior_scale=prior_scale[2])

		self.log_prior_dst = prior1d.log_prior_1d

		#================== R.A. prior ==========================
		if   prior[0] == "Uniform" :
			self.log_prior_ra  = self.Uniform
		elif prior[0] == "Gaussian" :
			self.log_prior_ra  = self.Gaussian		
		elif prior[0] == "Cauchy" :
			self.log_prior_ra  = self.Cauchy
		else:
			RuntimeError("Incorrect prior name")

		#================== Dec. prior ==========================
		if   prior[1] == "Uniform" :
			self.log_prior_dec  = self.Uniform
		elif prior[1] == "Gaussian" :
			self.log_prior_dec  = self.Gaussian		
		elif prior[1] == "Cauchy" :
			self.log_prior_dec  = self.Cauchy
		else:
			RuntimeError("Incorrect prior name")

		print("Posterior 3D initialized")

	######################### PRIORS #######################################
	def Uniform(self,theta,loc,scl):
		""" 
		Uniform prior
		"""
		return st.uniform.logpdf(theta,loc=loc-scl,scale=2*scl)

	def Gaussian(self,theta,loc,scl):
		"""
		Gaussian prior
		"""
		return st.norm.logpdf(theta,loc=loc,scale=scl)

	def Cauchy(self,theta,loc,scl):
		"""
		Cauchy prior
		"""
		return st.cauchy.logpdf(theta,loc=loc,scale=scl)

	#======== 3D prior ====================
	def log_prior_3d(self,theta):
		a = self.log_prior_ra(theta[0],self.prior_loc[0],self.prior_scl[0])
		b = self.log_prior_dec(theta[1],self.prior_loc[1],self.prior_scl[1])
		c = self.log_prior_dst(theta[2])
		return a+b+c
	#======================================

	#----------- Support -----------
	def Support(self,theta):
		if (theta[0] <   0.0 or theta[0] >= 360.0 or
		    theta[1] < -90.0 or theta[1] >   90.0 or
		    theta[2] <=  0.0) :
			return False
		else:
			return True

	def setup(self,datum):
		ra,dec,pax,u_ra,u_dec,u_pax,corr_ra_dec,corr_ra_pax,corr_dec_pax = datum
		corr      = np.zeros((self.ndim,self.ndim))
		corr[0,1] = corr_ra_dec
		corr[0,2] = corr_ra_pax
		corr[1,2] = corr_dec_pax

		corr      = corr + corr.T + np.eye(self.ndim)
		Sigma     = np.diag([u_ra,u_dec,u_pax]).dot(corr.dot(np.diag([u_ra,u_dec,u_pax])))
		Mu        = np.array([ra,dec,pax])
		
		self.corr_Mu   = Mu - self.zero_point
	

		try:
			self.inv = np.linalg.inv(Sigma)
		except Exception as e:
			sys.exit(e)
		else:
			pass
		finally:
			pass

		try:
			s,logdet = np.linalg.slogdet(Sigma)
		except Exception as e:
			sys.exit(e)
		else:
			pass
		finally:
			pass

		if s <= 0:
			sys.exit("Negative determinant!")

		self.cte  = -0.5*(np.log((2.0*np.pi)**3) + logdet)



	def __call__(self,theta):
		if not self.Support(theta):
			return -np.inf

		true_Mu   = np.array([theta[0],theta[1],1.0/theta[2]])

		x        = self.corr_Mu - true_Mu
		arg      = -0.5*np.dot(x.T,self.inv.dot(x))
		log_like = self.cte + arg
		log_posterior = self.log_prior_3d(theta) + log_like

		return log_posterior


