'''
Copyright 2020 Javier Olivares Romero

This file is part of Kalkayotl.

	Kalkayotl is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Kalkayotl is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Kalkayotl.  If not, see <http://www.gnu.org/licenses/>.
'''
#Afeter integrating the EFF kernel with r2dr -> (r^3 Hypergeometric2F1[3/2, g, 5/2, -r^2])/3
import sys
import numpy as np
import pandas as pd
from scipy.stats import rv_continuous
from scipy.optimize import root_scalar
from scipy.special import gamma as gamma_function
# import scipy.integrate as integrate
from scipy.special import hyp2f1
from scipy._lib._util import check_random_state
from scipy.stats._multivariate import _PSD, multi_rv_generic, multi_rv_frozen

def toCartesian(rho,dimension,random_state=None):
	size = rho.shape
	r = np.abs(rho)

	if dimension == 3:
		#http://corysimon.github.io/articles/uniformdistn-on-sphere/
		theta = np.arccos(1.-2*random_state.uniform(low=0.0,high=1.0,size=size))
		phi   = random_state.uniform(low=0.0,high=2*np.pi,size=size)
		x = r*np.sin(theta)*np.cos(phi)
		y = r*np.sin(theta)*np.sin(phi)
		z = r*np.cos(theta)
		samples = np.column_stack((x,y,z))

	elif dimension == 2:
		phi = random_state.uniform(low=-np.pi,high=np.pi,size=size)
		x = r*np.cos(phi)
		y = r*np.sin(phi)
		samples = np.column_stack((x,y))

	elif dimension == 1:
		samples = rho

	else:
		sys.exit("incorrect dimension")

	return samples

#=============== EFF generator ===============================
class EFF_1D(rv_continuous):
	"Univariate EFF distribution"
	""" This probability density function is defined for all x"""
	def _pdf(self,x,gamma):

		cte = np.sqrt(np.pi)*gamma_function(0.5*(gamma-1.))/gamma_function(gamma/2.)
		nx = (1. + x**2)**(-0.5*gamma)
		return nx/cte

	def _cdf(self,x,gamma):
		cte = np.sqrt(np.pi)*gamma_function(0.5*(gamma-1.))/gamma_function(gamma/2.)

		a = hyp2f1(0.5,0.5*gamma,1.5,-x**2)

		return 0.5 + x*(a/cte)
				

	def _rvs(self,gamma,size=1,random_state=None):
		#---------------------------------------------
		# Uniform between 0.01 and 0.99. It avoids problems with the
		# numeric integrator
		min_u = 1e-3
		us = np.random.uniform(min_u,1.-min_u,size=size).flatten()

		v = np.zeros_like(us)

		for i,u in enumerate(us):
			try:
				sol = root_scalar(
					lambda x : self._cdf(x,gamma=gamma) - u,
					xtol=1e-6,
					bracket=[0.0,1000.],
					method='bisect')
			except Exception as e:
				print(0.0,u,self._cdf(1000.0,gamma))
				raise
			v[i] = sol.root
			sol  = None
		return v.reshape(size)
#===============================================================

#=============== EFF generator ===============================
class EFF_1D_psd(rv_continuous):
	"Univariate EFF distribution"
	""" This probability density function is defined for 0<x<Rm"""
	def _pdf(self,x,gamma):
		y = 1 + self.b**2
		a = (1.0/(gamma-2.0))*(1.0-(y**(1.0-0.5*gamma))) # Truncated at Rm
		# a = 1.0/(gamma-2.0)                            #  No truncated
		nx = (1. + x**2)**(-0.5*gamma)
		return nx/a


	def _cdf(self,x,gamma):
		w  = 1. + x**2 
		y  = 1. + self.b**2
		a  = 1. - (w**(1.0-0.5*gamma))
		b  = 1. - (y**(1.0-0.5*gamma))
		return a/b
#===============================================================

#================== Multivariate EFF =============================
"""
Inspired by the Multivariate t-distribution of Gregory Gundersen 2020, 
based of the architecture ofSciPy's
`_multivariate.py` module by Joris Vankerschaver 2013.
"""

class multivariate_eff_gen(multi_rv_generic):

	def __init__(self, seed=None):
		"""Initialize a multivariate EFF random variable.

		Parameters
		----------
		seed : Random state.
		"""
		self._random_state = check_random_state(seed)

	def __call__(self, loc=None, scale=1, gamma=3, seed=None):
		"""Create a frozen multivariate EFF distribution See
		`multivariate_eff_frozen` for parameters.
		"""
		return multivariate_eff_frozen(loc=loc, scale=scale, gamma=gamma, seed=seed)

	def pdf(self, x, loc=None, scale=1, gamma=3):
		"""Multivariate EFF probability density function.

		Parameters
		----------
		x : array_like
			Points at which to evaluate the log of the probability density
			function.
		loc : array_like, optional
			Mean of the distribution (default zero).
		scale : array_like, optional
			Positive definite shape matrix. This is not the distribution's
			covariance matrix (default one).
		gamma : EFF's gamma parameter.

		Returns
		-------
		logpdf : Probability density function evaluated at `x`.

		"""
		lp = self.logpdf(x, loc, scale, gamma)
		return np.exp(lp)

	def logpdf(self, x, loc=None, scale=1, gamma=3):
		"""Log of the multivariate EFF probability density function.

		Parameters
		----------
		x : array_like
			Points at which to evaluate the log of the probability density
			function.
		loc : array_like, optional
			Centre of the distribution (default zero).
		scale : array_like, optional
			Positive definite shape matrix. This is not the distribution's
			covariance matrix (default one).
		gamma : EFF's gamma parameter.

		Returns
		-------
		logpdf : Log of the probability density function evaluated at `x`.

		"""
		dim, loc, scale, gamma = self._process_parameters(loc, scale, gamma)
		assert dim == 3, "Only valid for 3D!"
		x = self._process_quantiles(x, dim)
		scale_info = _PSD(scale)
		
		return self._logpdf(x,loc,scale_info.U,scale_info.log_pdet,gamma,dim)

	def _logpdf(self, x, loc, U, log_pdet, gamma,dim):
		# U is inverse Cholesky, so only one product U*x instead of sqrt(x*U*U.T*x)
		quaddist = np.square(np.dot(x - loc,U)).sum(axis=-1)

		a = 1.5*np.log(np.pi)
		b = np.log(gamma_function(0.5*(gamma-3.)))
		c = np.log(gamma_function(0.5*gamma))
		log_cte = a + b -c
		log_d = -0.5*gamma*np.log1p(quaddist) - log_cte - 3.*log_pdet
		
		return log_d

	def _cdf(self, x, rc, gamma):
		# use only to generate random variates
		a = x**2 + rc**2
		hg = gamma_function(0.5*gamma)
		hg3 = gamma_function(0.5*(gamma-3.))
		hyp = hyp2f1(-0.5,0.5*gamma,0.5,-(x/rc)**2)
		b = -(rc**gamma)*a*(rc**2 + (gamma-1.)*(x**2))
		c = (rc**4)*(a**(0.5*gamma))*hyp

		up = 4.*(a**(-0.5*gamma))*hg*( b + c)
		down = (gamma-3.)*(gamma-1.)*np.sqrt(np.pi)*(rc**3)*x*hg3
		result = up/down
		return result

	def _rvs(self, rc,gamma, size,upper=[0.99,1e6]):
		#------ This rvs are standardized and are not meant to be used
		#------- but in combination with rvs
		#---------------------------------------------
		# Uniform between 0.0 and upper
		us = self._random_state.uniform(0.0,upper[0],size=size).flatten()

		v = np.zeros_like(us)

		for i,u in enumerate(us):
			try:
				sol = root_scalar(
							lambda x : self._cdf(x,rc,gamma) - u,
							xtol=1e-6,
							bracket=[1e-6,upper[1]],
							method='bisect')
			except Exception as e:
				print("Value outside range [{0},{1},{2}]".format(self._cdf(0.0,rc,gamma),
					u,self._cdf(upper[1],rc,gamma)))
				raise
			v[i] = sol.root
			sol  = None
		return v.reshape(size)

	def rvs(self, loc=None, scale=1, gamma=3, size=1, random_state=None):
		"""Draw random samples from a multivariate EFF distribution.

		Parameters
		----------
		x : array_like
			Points at which to evaluate the log of the probability density
			function.
		loc : array_like, optional
			Mean of the distribution (default zero).
		scale : array_like, optional
			Positive definite shape matrix. This is not the distribution's
			covariance matrix (default one).
		gamma : EFF gamma parameter.

		Returns
		-------
		"""
		
		if random_state is not None:
			rng = check_random_state(random_state)
		else:
			rng = self._random_state
		dim, loc, scale, gamma = self._process_parameters(loc, scale, gamma)
		assert dim == 3, "Only valid for 3D!"
		scale_info = _PSD(scale)

		#------ Take samples from the distance -------------
		rho = self._rvs(rc=1.0, gamma=gamma, size=size)

		#------ Samples from the angles -------
		samples = toCartesian(rho,dim,random_state=rng).reshape(size,dim)

		chol = np.linalg.cholesky(scale)
		samples = np.dot(samples,chol)

		return loc + samples


	def _process_quantiles(self, x, dim):
		"""Adjust quantiles array so that last axis labels the components of
		each data point.
		"""
		x = np.asarray(x, dtype=float)
		if x.ndim == 0:
			x = x[np.newaxis]
		elif x.ndim == 1:
			if dim == 1:
				x = x[:, np.newaxis]
			else:
				x = x[np.newaxis, :]
		return x

	def _process_parameters(self, loc, scale, gamma):
		"""Infer dimensionality from mean array and shape matrix, handle
		defaults, and ensure compatible dimensions.
		"""
		if loc is None and scale is None:
			scale = np.asarray(1, dtype=float)
			dim = 1
		elif loc is None:
			scale = np.asarray(scale, dtype=float)
			if scale.ndim < 2:
				dim = 1
			else:
				dim = scale.shape[0]
			loc = np.zeros(dim)
		else:
			scale = np.asarray(scale, dtype=float)
			loc = np.asarray(loc, dtype=float)
			dim = loc.size

		#-------- Correct shape in case of 1D ----------
		if dim == 1:
			loc.shape = (1,)
			scale.shape = (1, 1)

		if loc.ndim != 1 or loc.shape[0] != dim:
			raise ValueError("Array 'loc' must be a vector of length %d." %
							 dim)
		if scale.ndim == 0:
			scale = scale * np.eye(dim)
		elif scale.ndim == 1:
			scale = np.diag(scale)
		elif scale.ndim == 2 and scale.shape != (dim, dim):
			rows, cols = scale.shape
			if rows != cols:
				msg = ("Array 'scale' must be square if it is two dimensional,"
					   " but scale.shape = %s." % str(scale.shape))
			else:
				msg = ("Dimension mismatch: array 'scale' is of shape %s,"
					   " but 'mean' is a vector of length %d.")
				msg = msg % (str(scale.shape), len(loc))
			raise ValueError(msg)
		elif scale.ndim > 2:
			raise ValueError("Array 'scale' must be at most two-dimensional,"
							 " but scale.ndim = %d" % scale.ndim)

		# Process gamma.
		if gamma is None:
			gamma = dim
		if np.any(gamma < dim):
			raise ValueError("Gamma must be > dim, but it is %d"%gamma)

		return dim, loc, scale, gamma


class multivariate_eff_frozen(multi_rv_frozen):

	def __init__(self, loc=None, scale=1, gamma=2, seed=None):
		"""
		Create a frozen multivariate EFF frozen distribution.

		Parameters
		----------
		x : array_like
			Points at which to evaluate the log of the probability density
			function.
		loc : array_like, optional
			Mean of the distribution (default zero).
		scale : array_like, optional
			Positive definite shape matrix. This is not the distribution's
			covariance matrix (default one).
		gamma : EFF gamma parameter.

		"""
		self._dist = multivariate_eff_gen(seed)
		self.dim, self.loc, self.scale, self.gamma = self._dist._process_parameters(loc, scale, gamma)
		assert self.dim == 3, "Only valid for 3D!"
		self.scale_info = _PSD(self.scale)

	def logpdf(self, x):
		x = self._dist._process_quantiles(x, self.dim)

		return self._dist._logpdf(x, self.loc, self.scale_info.U, self.scale_info.log_pdet, self.gamma, self.dim)


	def pdf(self, x):
		return np.exp(self.logpdf(x))

	def cdf(self, x,rc,gamma):
		return self._dist._cdf(x,rc,gamma)

	def rvs(self, size=1, random_state=None):
		"""
		Draw random samples from a multivariate EFF distribution.

		Parameters
		----------
		size : integer, optional
			Number of samples to draw (default 1).
		random_state : np.random.RandomState instance
			RandomState used for drawing the random variates.

		Returns
		-------
		rvs : ndarray or scalar
			Random variates of size (`size`, `N`), where `N` is the
			dimension of the random variable.
		"""
		return self._dist.rvs(loc=self.loc,
							  scale=self.scale,
							  gamma=self.gamma,
							  size=size,
							  random_state=random_state)

mveff = multivariate_eff_gen()


###################################################### TEST ################################################################################
import sys
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as st

def test_eff_1d_psd(n=1000,rc=1.,gamma=5.0,mc_file=None,bins=40):
	# ----- Generate samples ---------
	eff = EFF_1D_psd(name='EFF',a=0.0,b=20.0)
	r = eff.rvs(scale=rc,gamma=gamma,size=n)

	if mc_file is not None:
		mc = pd.read_csv(mc_file,sep="\t",header=None,skiprows=1,
					names=["Mass","X","Y","Z","U","V","W"])

		pos = mc[["X","Y","Z"]].to_numpy()

		s = np.sqrt(np.sum(pos**2,axis=1))

	#------ grid ----------------------
	xlim = (0.0,30*rc)
	x = np.linspace(xlim[0],xlim[1],1000)
	y = eff.pdf(x,scale=rc,gamma=gamma)
	z = eff.cdf(x,scale=rc,gamma=gamma)

	pdf = PdfPages(filename="Test_EFF.pdf")
	plt.figure()
	plt.hist(r,bins=bins,density=True,histtype='step',label="EFF_1D_psd")
	if mc_file is not None:
		plt.hist(s,bins=bins,density=True,histtype='step',label="McCluster")
	plt.yscale('log')
	plt.legend()
	pdf.savefig(bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.plot(x,y,color="black",label="PDF")
	plt.yscale('log')
	plt.legend()
	pdf.savefig(bbox_inches='tight')
	plt.close()

	plt.figure(1)
	plt.plot(x,z,color="black",label="CDF")
	plt.yscale('log')
	plt.legend()
	pdf.savefig(bbox_inches='tight')
	plt.close(1)
	
	pdf.close()

def test_eff_3d(n=1000,r0=0.,rc=1.,gamma=5.0,mc_file=None):
	loc = np.array([r0,0.0,0.0])
	scl = np.array([[rc,0.0,0.0],
					[0.0,rc,0.0],
					[0.0,0.0,rc]])

	if mc_file is not None:
		mc = pd.read_csv(mc_file,sep="\t",header=None,skiprows=1,
					names=["Mass","X","Y","Z","U","V","W"])
		s_mc = mc[["X","Y","Z"]].to_numpy()
		r_mc = np.sqrt(np.sum(s_mc**2,axis=1))

	# ----- Generate samples ---------
	ue = EFF_1D_psd(name='EFF',a=0.0,b=20.0)
	me = mveff(loc=loc,scale=scl,gamma=gamma)
	s_mv = me.rvs(size=n)
	r_mv = np.sqrt(np.sum(s_mv**2,axis=1))
	# print(s_mv)

	#-------- Density ---------------
	z_mv = me.pdf(s_mv)
	z_mc = me.pdf(s_mc)
	z_mv_1d = ue.pdf(r_mv,scale=rc,gamma=gamma)
	z_mc_1d = ue.pdf(r_mc,scale=rc,gamma=gamma)

	#------ grid ----------------------
	range_dist = (0,10)
	x = np.linspace(0.0,20.0,100)
	y = ue.cdf(x,scale=rc,gamma=gamma)	
	c = me.cdf(x,rc=rc, gamma=gamma)

	pdf = PdfPages(filename="Test_EFF_3D.pdf")

	#-------------- Distance -------------------
	plt.figure(0)
	plt.hist(r_mv,bins=100,density=True,histtype='step',label="Samples")
	plt.hist(r_mc,bins=100,density=True,histtype='step',label="McCluster")
	plt.legend()
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	plt.figure(0)
	plt.plot(x,c,label="3D CDF")
	plt.plot(x,y,label="1D CDF")
	plt.legend()
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	plt.figure(0)
	plt.scatter(r_mc,z_mc,marker="s",label="McCluster")
	plt.scatter(r_mv,z_mv,marker=".",label="Samples")
	plt.scatter(r_mc,z_mc_1d,marker="s",label="McCluster 1D")
	plt.scatter(r_mv,z_mv_1d,marker=".",label="Samples 1D")
	plt.legend()
	plt.xlabel("R")
	plt.ylabel("Density")
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	plt.figure(0)
	plt.scatter(s_mv[:,0],s_mv[:,1],s=1,c=z_mv,marker=".",label="Samples")
	plt.scatter(s_mc[:,0],s_mc[:,1],s=1,c=z_mc,marker="s",label="McCluster")
	plt.legend()
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.gca().set_aspect('equal', adjustable='box')
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	plt.figure(0)
	plt.scatter(s_mv[:,2],s_mv[:,0],s=1,c=z_mv,marker=".",label="Samples")
	plt.scatter(s_mc[:,2],s_mc[:,0],s=1,c=z_mc,marker="s",label="McCluster")
	plt.legend()
	plt.xlabel("X")
	plt.ylabel("Z")
	plt.gca().set_aspect('equal', adjustable='box')
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	#-------------- Distance -------------------
	plt.figure(0)
	plt.hist(s_mv[:,0],bins=100,density=True,histtype='step',label="X Samples")
	plt.hist(s_mv[:,1],bins=100,density=True,histtype='step',label="Y Samples")
	plt.hist(s_mv[:,2],bins=100,density=True,histtype='step',label="Z Samples")
	plt.hist(s_mc[:,0],bins=100,density=True,histtype='step',label="X McCluster")
	plt.hist(s_mc[:,1],bins=100,density=True,histtype='step',label="Y McCluster")
	plt.hist(s_mc[:,2],bins=100,density=True,histtype='step',label="Z McCluster")
	plt.yscale('log')
	plt.ylim(bottom=1e-5)
	plt.legend()
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	
	pdf.close()

if __name__ == "__main__":

	mc_file = "/home/jolivares/Repos/Amasijo/Data/EFF_n1000_r1_g5.txt"
	# test_eff_1d_psd(n=1000,rc=1.,gamma=5,mc_file=mc_file)
	test_eff_3d(mc_file=mc_file)