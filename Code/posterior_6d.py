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
from posterior_5d import Posterior as posterior_5d

class Posterior:
	"""
	This class provides flexibility to infer the posterior distribution of the 6D
	"""
	def __init__(self,prior="Uniform",prior_loc=0,prior_scale=100,
		zero_point=[0.0,0.0,0.0,0.0,0.0,0.0]):

		self.ndim        = 6
		self.prior_loc   = prior_loc
		self.prior_scl   = prior_scale
		self.zero_point  = zero_point

		#================= 5D Prior ===========================
		prior5d = posterior_5d(prior=prior,prior_loc=prior_loc,
								prior_scale=prior_scale)

		self.log_prior_5d = prior5d.log_prior_5d

		print("Posterior 6D initialized")

	#======== 6D prior ====================
	def log_prior_6d(self,theta):
		lp_5d    = self.log_prior_5d(theta[:5])
		lp_rvel  = st.cauchy.logpdf(theta[5],loc=0.0,scale=300.0) #km*s**-1
		return lp_5d + lp_rvel
	#======================================

	#----------- Support -----------
	def Support(self,theta):
		# No restrictions on the value of true proper motions or radial_velocity
		if (theta[0] <   0.0 or theta[0] >= 360.0 or
		    theta[1] < -90.0 or theta[1] >   90.0 or
		    theta[2] <=  0.0) :
			return False
		else:
			return True

	def setup(self,datum):
		# ra,dec,pax,pmra,pmdec,rvel                            = datum[:6]
		# u_ra,u_dec,u_pax,u_pmra,u_pmdec,u_rvel                = datum[6:12]
		# ra_dec_corr,ra_pax_corr,ra_pmra_corr,ra_pmdec_corr    = datum[12:16]
		# dec_pax_corr,dec_pmra_corr,dec_pmdec_corr             = datum[16:19]
		# pax_pmra_corr,pax_pmdec_corr                          = datum[19:21]
		# pmra_pmdec_corr                                       = datum[21]
		Mu = np.array(datum[:6])
		Un = np.array(datum[6:12])		

		corr       = np.zeros((self.ndim,self.ndim))
		corr[0,1:5] = datum[12:16]
		corr[1,2:5] = datum[16:19]
		corr[2,3:5] = datum[19:21]
		corr[3,4]   = datum[21]

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

		self.cte  = -0.5*(np.log((2.0*np.pi)**6) + logdet)



	def __call__(self,theta):
		if not self.Support(theta):
			return -np.inf

		true_Mu    = np.array(theta)
		true_Mu[2] = 1.0/theta[2]

		x        = self.corr_Mu - true_Mu
		arg      = -0.5*np.dot(x.T,self.inv.dot(x))
		log_like = self.cte + arg
		log_posterior = self.log_prior_6d(theta) + log_like

		return log_posterior


