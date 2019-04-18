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
from posterior_1d import Posterior as posterior_1d

class Posterior:
	"""
	This class provides flexibility to infer the posterior distribution of the 3D
	"""
	def __init__(self,prior="Uniform",prior_loc=0,prior_scale=100,zero_point=[0.0,0.0,0.0]):

		self.ndim        = 3
		self.prior_loc   = prior_loc
		self.prior_scl   = prior_scale
		self.zero_point  = zero_point

		#================= 1D Prior ===========================

		prior1d = posterior_1d(prior=prior,prior_loc=prior_loc,
								prior_scale=prior_scale)

		self.log_prior_1d = prior1d.log_prior_1d

		#================== 2D prior ==========================
		prior_ra   = 1.0/360

		prior_de   = 1.0/180

		self.log_prior_2d = np.log(prior_ra) + np.log(prior_de)
		print("Posterior 3D initialized")

	#======== 3D prior ====================
	def log_prior_3d(self,theta):
		a = self.log_prior_2d
		b = self.log_prior_1d(theta[2])
		return a+b
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


