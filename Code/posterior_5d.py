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
from posterior_3d import Posterior as posterior_3d

class Posterior:
	"""
	This class provides flexibility to infer the posterior distribution of the 5D
	"""
	def __init__(self,prior="Uniform",prior_loc=0,prior_scale=100,
		zero_point=[0.0,0.0,0.0,0.0,0.0]):

		self.ndim        = 5
		self.prior_loc   = prior_loc
		self.prior_scl   = prior_scale
		self.zero_point  = zero_point

		#================= 3D Prior ===========================
		prior3d = posterior_3d(prior=prior,prior_loc=prior_loc,
								prior_scale=prior_scale)

		self.log_prior_3d = prior3d.log_prior_3d

		print("Posterior 5D initialized")

	#======== 5D prior ====================
	def log_prior_5d(self,theta):
		lp_3d    = self.log_prior_3d(theta[:3])
		lp_pmra  = st.cauchy.logpdf(theta[3],loc=0.0,scale=500.0) #mas*yr**-1
		lp_pmdec = st.cauchy.logpdf(theta[4],loc=0.0,scale=500.0) #mas*yr**-1
		return lp_3d + lp_pmra + lp_pmdec
	#======================================

	#----------- Support -----------
	def Support(self,theta):
		# No restrictions on the value of true proper motions
		if (theta[0] <   0.0 or theta[0] >= 360.0 or
		    theta[1] < -90.0 or theta[1] >   90.0 or
		    theta[2] <=  0.0) :
			return False
		else:
			return True

	def setup(self,datum):
		# ra,dec,pax,pmra,pmdec                                 = datum[:5]
		# u_ra,u_dec,u_pax,u_pmra,u_pmdec                       = datum[5:10]
		# ra_dec_corr,ra_pax_corr,ra_pmra_corr,ra_pmdec_corr    = datum[10:14]
		# dec_pax_corr,dec_pmra_corr,dec_pmdec_corr             = datum[14:17]
		# pax_pmra_corr,pax_pmdec_corr                          = datum[17:19]
		# pmra_pmdec_corr                                       = datum[19]
		Mu = np.array(datum[:5])
		Un = np.array(datum[5:10])		

		corr       = np.zeros((self.ndim,self.ndim))
		corr[0,1:] = datum[10:14]
		corr[1,2:] = datum[14:17]
		corr[2,3:] = datum[17:19]
		corr[3,4]  = datum[19]

		corr      = corr + corr.T + np.eye(self.ndim)
		Sigma     = np.diag(Un).dot(corr.dot(np.diag(Un)))
		
		self.corr_Mu   = Mu + self.zero_point
	

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

		self.cte  = -0.5*(np.log((2.0*np.pi)**5) + logdet)



	def __call__(self,theta):
		if not self.Support(theta):
			return -np.inf

		true_Mu    = np.array(theta)
		true_Mu[2] = 1.0/theta[2]

		x        = self.corr_Mu - true_Mu
		arg      = -0.5*np.dot(x.T,self.inv.dot(x))
		log_like = self.cte + arg
		log_posterior = self.log_prior_5d(theta) + log_like

		return log_posterior


