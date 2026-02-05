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

import sys
import numpy as np
from scipy.stats import rv_continuous
from scipy.optimize import root_scalar
from scipy.special import gamma as gamma_function
import scipy.integrate as integrate
from scipy.special import hyp2f1
from scipy._lib._util import check_random_state
from scipy.stats._multivariate import _PSD, multi_rv_generic, multi_rv_frozen

__all__ = [
	"toCartesian",
	"edsd",
	"king"
]

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


#=============== 1D distributions ===============================================
class edsd_gen(rv_continuous):
	"EDSD distribution"
	def _pdf(self, x,L):
		return (0.5 * ( x**2 / L**3)) * np.exp(-x/L)

	def _cdf(self, x,L):
		result = 1.0 - np.exp(-x/L)*(x**2 + 2. * x * L + 2. * L**2)/(2.*L**2)
		return result

	def _rvs(self,L):
		sz, rndm = self._size, self._random_state
		u = rndm.random_sample(size=sz)

		v = np.zeros_like(u)

		for i in range(sz[0]):

			sol = root_scalar(lambda x : self._cdf(x,L) - u[i],
				bracket=[0,1.e10],
				method='brentq')
			v[i] = sol.root
		return v



edsd = edsd_gen(a=0.0,name='edsd')

#=============== GDD generator ===============================
class ggd_gen(rv_continuous):
	"GGD distribution"
	def _pdf(self, x,L,alpha,beta):
		fac1 = 1.0 / gamma_function((beta+1.0)/alpha)
		fac2 = alpha / np.power(L, beta+1.0)
		fac3 = np.power(r, beta)
		fac4 = np.exp(-np.power(r/L, alpha))
		return fac1*fac2*fac3*fac4

	def _cdf(self, x,L,alpha,beta):
		result = gamma_incomplete((beta+1.0)/alpha,np.power(r/L,alpha))
		return result

	def _rvs(self,L,alpha,beta):
		sz, rndm = self._size, self._random_state
		u = rndm.random_sample(size=sz)

		v = np.zeros_like(u)

		for i in range(sz[0]):

			sol = root_scalar(lambda x : self._cdf(x,L,alpha,beta) - u[i],
				bracket=[0,1.e10],
				method='brentq')
			v[i] = sol.root
		return v
    
    
ggd = ggd_gen(name='ggd')

#=============== King generator ===============================
class king_gen(rv_continuous):
	"King distribution"
	def _pdf(self,x,rt):
		"""
		The tidal radius is in units of the core radius
		"""
		a = 1. + rt**2
		b = rt/a
		c = 2.*np.arcsinh(rt)/np.sqrt(a)
		d = np.arctan(rt)
		cte = 2.*( b - c + d)

		u = 1./np.sqrt(1. +  x**2)
		v = 1./np.sqrt(a)

		res = ((u-v)**2)/cte

		return np.where(np.abs(x) < rt,res,np.full_like(x,np.nan))

	def _cdf(self,x,rt):
		u   = 1 + rt**2
		cte = 2.*( rt/(1 + rt**2) - 2.*np.arcsinh(rt)/np.sqrt(1.+ rt**2) + np.arctan(rt))

		val = (rt+x)/u - (2.*(np.arcsinh(rt)+np.arcsinh(x))/np.sqrt(u)) + (np.arctan(rt)+np.arctan(x))

		res = val/cte
		
		return res
				

	def _rvs(self,rt,size=1,random_state=None):
		#----------------------------------------
		min_u = 1e-3
		us = np.random.uniform(min_u,1.-min_u,size=size).flatten()

		v = np.zeros_like(us)

		for i,u in enumerate(us):
			try:
				sol = root_scalar(lambda x : self._cdf(x,rt) - u,
				bracket=[-rt,rt],
				method='brentq')
			except Exception as e:
				print(u[i])
				print(self._cdf(-rt,rt))
				print(self._cdf(rt,rt))
				raise
			v[i] = sol.root
			sol  = None
		return v.reshape(size)

king = king_gen(name='King')

#============= Multivariate King ============================
"""
Inspired by the Multivariate t-distribution of Gregory Gundersen 2020, 
based of the architecture ofSciPy's
`_multivariate.py` module by Joris Vankerschaver 2013.
"""

class multivariate_king_gen(multi_rv_generic):

	def __init__(self, seed=None):
		"""Initialize a multivariate EFF random variable.

		Parameters
		----------
		seed : Random state.
		"""
		self._random_state = check_random_state(seed)

	def __call__(self, loc=None, scale=1, rt=10, seed=None):
		"""Create a frozen multivariate EFF distribution See
		`multivariate_eff_frozen` for parameters.
		"""
		return multivariate_king_frozen(loc=loc, scale=scale, rt=rt, seed=seed)

	def pdf(self, x, loc=None, scale=1, rt=10):
		"""Multivariate King probability density function.

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
		rt : King's tidal radius parameter.

		Returns
		-------
		logpdf : Probability density function evaluated at `x`.

		"""
		lp = self.logpdf(x, loc, scale, rt)
		return np.exp(lp)

	def logpdf(self, x, loc=None, scale=1, rt=10):
		"""Log of the multivariate King probability density function.

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
		rt : King's tidal radius parameter.

		Returns
		-------
		logpdf : Log of the probability density function evaluated at `x`.

		"""
		dim, loc, scale, rt = self._process_parameters(loc, scale, rt)
		x = self._process_quantiles(x, dim)
		scale_info = _PSD(scale)
		if dim == 1:
			return self._logpdf1d(x,loc,scale_info.U,scale_info.log_pdet,rt)
		elif dim == 3:
			return self._logpdf3d(x,loc,scale_info.U,scale_info.log_pdet,rt)
		else:
			sys.exit("Dimension not implemented")

	def _logpdf1d(self, x, loc, U, log_pdet, rt):
		# U is inverse Cholesky, so only one product U*x instead of sqrt(x*U*U.T*x)
		#------ These assume that the distance is scaled
		rc = np.exp(log_pdet)
		quaddist = ((x-loc)/rc)**2

		a = 1. + rt**2
		b = rt/a
		c = 2.*np.arcsinh(rt)/np.sqrt(a)
		d = np.arctan(rt)
		cte = 2*( b - c + d )

		u = 1./np.sqrt(1. + quaddist)
		v = 1./np.sqrt(a)

		res = 2.*np.log(u-v) - np.log(cte) - log_pdet # Here we introduce the scale
		#Notice that this is similar to pdf((x-loc)/scl)/scl

		return np.where(np.sqrt(quaddist) < rt,res,np.full_like(d,np.nan))

	def _logpdf3d(self, x, loc, U, log_pdet, rt):
		# U is inverse Cholesky, so only one product U*x instead of sqrt(x*U*U.T*x)
		quaddist = np.square(np.dot(x - loc,U)).sum(axis=-1)

		# Remember rc = np.exp(log_pdet)
		#--- The following assumes rc = 1

		a = 1. + rt**2
		b = (rt**3)/(3.*a)
		c = np.arcsinh(rt)/np.sqrt(a)
		d = np.arctan(rt)

		cte = 4.*np.pi*(b+c-d)

		u = 1./np.sqrt(1. + quaddist)
		v = 1./np.sqrt(a)
		res = 2.*np.log(u-v) - np.log(cte) - 3*log_pdet # Here we introduce the scale rc**3

		return np.where(np.sqrt(quaddist) < rt,res,np.full_like(quaddist,np.nan))

	def _cdf1d(self, x, rt):
		#------ This CDF is standardized and should not be used other that 
		#------ for generating rvs
		a   = 1. + rt**2
		b = rt/a
		c = 2.*np.arcsinh(rt)/np.sqrt(a)
		d = np.arctan(rt)

		e = x/a
		f = 2.*(np.arcsinh(rt)+np.arcsinh(x))/np.sqrt(a)
		g = np.arctan(rt)+np.arctan(x)

		res = (b + e - f + g)/(2*( b - c + d ))
		
		return np.where(x < rt, res, np.full_like(x,np.nan))


	def _cdf3d(self, x, rt):
		#------ This CDF is standardized and should not be used other that 
		#------ for generating rvs
		a = 1.+rt**2
		b = (rt**3)/(3.*a)
		c = np.arcsinh(rt)/np.sqrt(a)
		d = np.arctan(rt)

		e = (x**3)/(3.*a)
		f = (-x*np.sqrt(1+x**2)+np.arcsinh(x))/np.sqrt(a)
		g = np.arctan(x)

		res = (x + e + f - g)/(b+c-d)

		return np.where(x < rt, res, np.full_like(x,np.nan))

	def _rvs_r1d(self,rt,size,min_u):
		#------ This rvs are standardized and are not meant to be used
		#------- but in combination with rvs
		#---------------------------------------------
		# Uniform between 0.01 and 0.99. It avoids problems with the
		# numeric integrator
		us = self._random_state.uniform(min_u,1.-min_u,size=size).flatten()

		v = np.zeros_like(us)

		for i,u in enumerate(us):
			try:
				sol = root_scalar(
					lambda x : self._cdf1d(x,rt) - u,
					xtol=1e-6,
					bracket=[-rt,rt],
					method='bisect')
			except Exception as e:
				print("Value outside range!")
				raise
			v[i] = sol.root
			sol  = None
		return v.reshape(size)

	def _rvs_r3d(self, rt, size,min_u):
		#------ This rvs are standardized and are not meant to be used
		#------- but in combination with rvs
		#---------------------------------------------
		# Uniform between 0.01 and 0.99. It avoids problems with the
		# numeric integrator
		us = self._random_state.uniform(min_u,1.-min_u,size=size).flatten()

		v = np.zeros_like(us)

		for i,u in enumerate(us):
			try:
				sol = root_scalar(
							lambda x : self._cdf3d(x, rt) - u,
							xtol=1e-6,
							bracket=[0.0,rt],
							method='bisect')
			except Exception as e:
				print("Value outside range")
				raise
			v[i] = sol.root
			sol  = None
		return v.reshape(size)
		

	def rvs(self, loc=None, scale=1, rt=10, size=1, random_state=None,
				min_u=1e-5):
		"""Draw random samples from a multivariate King distribution.

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
		rt : King's tidal radius parameter.

		Returns
		-------
		"""
		
		if random_state is not None:
			rng = check_random_state(random_state)
		else:
			rng = self._random_state

		dim, loc, scale, rt = self._process_parameters(loc, scale, rt)
		scale_info = _PSD(scale)
		# rt = rt*np.exp(0.5*scale_info.log_pdet)

		#------ Take samples from the distance -------------
		if dim == 1 :
			rho = self._rvs_r1d(rt=rt, size=size,min_u=min_u)
		elif dim == 3:
			rho = self._rvs_r3d(rt=rt, size=size,min_u=min_u)
		else:
			sys.exit("Dimension {0} not implemented".format(dim))

		#------ Samples from the angles -------
		samples = toCartesian(rho,dim,random_state=rng).reshape(size,dim)

		if dim > 1:
			chol = np.linalg.cholesky(scale)
			samples = np.dot(samples,chol)
		else:
			samples = scale*samples

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

	def _process_parameters(self, loc, scale, rt):
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

		# Process rt.
		if rt is None:
			rt = 10.
		if np.any(rt <= 1.):
			raise ValueError("rt >= 1, but it is %d"%rt)

		return dim, loc, scale, rt


class multivariate_king_frozen(multi_rv_frozen):

	def __init__(self, loc=None, scale=1, rt=10, seed=None):
		"""
		Create a frozen multivariate King frozen distribution.

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
		rt : King's tidal radius parameter.

		"""
		self._dist = multivariate_king_gen(seed)
		self.dim, self.loc, self.scale, self.rt = self._dist._process_parameters(loc, scale, rt)
		self.scale_info = _PSD(self.scale)

	def logpdf(self, x):
		x = self._dist._process_quantiles(x, self.dim)
		if self.dim == 1:
			return self._dist._logpdf1d(x,self.loc,self.scale_info.U,self.scale_info.log_pdet,self.rt)
		elif self.dim == 3:
			return self._dist._logpdf3d(x,self.loc,self.scale_info.U,self.scale_info.log_pdet,self.rt)
		else:
			sys.exit("Dimension not implemented")

	def pdf(self, x):
		return np.exp(self.logpdf(x))

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
							  rt=self.rt,
							  size=size,
							  random_state=random_state)

mvking = multivariate_king_gen()



###################################################### TEST ################################################################################
import sys
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as st


def test_edsd(n=1000,L=100.):
	#----- Generate samples ---------
	s = edsd.rvs(L=L,size=n)


	#------ grid -----
	x = np.linspace(0,10.*L,100)
	y = edsd.pdf(x,L=L)
	z = (x**2/(2.*L**3))*np.exp(-x/L)

	pdf = PdfPages(filename="Test_EDSD.pdf")
	plt.figure(1)
	plt.hist(s,bins=50,density=True,color="grey",label="Samples")
	plt.plot(x,y,color="black",label="PDF")
	plt.plot(x,z,color="red",linestyle="--",label="True")
	plt.legend()
	
	#-------------- Save fig --------------------------
	pdf.savefig(bbox_inches='tight')
	plt.close(1)
	
	pdf.close()
    
def test_ggd(n=1000,L=100.,alpha=1.,beta=2.):
	#----- Generate samples ---------
	s = ggd.rvs(L=L,alpha=alpha,beta=beta,size=n)

	#------ grid -----
	x = np.linspace(0,10.*L,100)
	y = ggd.pdf(x,L=L,alpha=alpha,beta=beta)
	fac1 = 1.0 / gamma_function((beta+1.0)/alpha)
	fac2 = alpha / np.power(L, beta+1.0)
	fac3 = np.power(r, beta)
	fac4 = np.exp(-np.power(x/L, alpha))
	z = fac1*fac2*fac3*fac4

	pdf = PdfPages(filename="Test_GGD.pdf")
	plt.figure(1)
	plt.hist(s,bins=50,density=True,color="grey",label="Samples")
	plt.plot(x,y,color="black",label="PDF")
	plt.plot(x,z,color="red",linestyle="--",label="True")
	plt.legend()
	
	#-------------- Save fig --------------------------
	pdf.savefig(bbox_inches='tight')
	plt.close(1)
	
	pdf.close()



def test_king(n=100000,r0=0.,rc=2.,rt=20.):
	#----- Generate samples ---------
	s = king.rvs(loc=r0,scale=rc,rt=rt/rc,size=n)
	#------ grid -----
	
	range_dist = (r0-1.5*rt,r0+1.5*rt)
	x = np.linspace(range_dist[0],range_dist[1],1000)
	y = king.pdf(x,loc=r0,scale=rc,rt=rt/rc)
	z = king.cdf(x,loc=r0,scale=rc,rt=rt/rc)
	
	pdf = PdfPages(filename="Test_King.pdf")
	plt.figure(0)
	plt.hist(s,bins=100,range=range_dist,density=True,color="grey",label="Samples")
	plt.plot(x,y,color="black",label="PDF")
	plt.xlim(range_dist)
	plt.yscale('log')
	plt.legend()
	
	#-------------- Save fig --------------------------
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	plt.figure(1)
	plt.plot(x,z,color="black",label="CDF")
	plt.xlim(range_dist)
	plt.legend()
	
	#-------------- Save fig --------------------------
	pdf.savefig(bbox_inches='tight')
	plt.close(1)
	
	pdf.close()



def test_king_1d(n=10000,r0=0.,rc=1.,rt=10):
	# ----- Generate samples ---------
	s_un = king.rvs(loc=r0,scale=rc,rt=rt,size=n)
	s_mv = mvking.rvs(loc=r0,scale=rc,rt=rt,size=n)

	#------ grid ----------------------
	range_dist = (r0-20*rc,r0+20*rc)
	x = np.linspace(range_dist[0],range_dist[1],1000)
	y_un = king.pdf(x,loc=r0,scale=rc,rt=rt)
	y_mv = mvking.pdf(x,loc=r0,scale=rc,rt=rt)
	z_un = king.cdf(x,loc=r0,scale=rc,rt=rt)

	pdf = PdfPages(filename="Test_King_1D.pdf")
	plt.figure(0)
	plt.hist(s_un,bins=100,range=range_dist,density=True,histtype='step',color="grey",label="Univariate")
	plt.hist(s_mv,bins=100,range=range_dist,density=True,histtype='step',color="green",label="Multivariate 1D")
	plt.plot(x,y_un,color="black",label="Univariate PDF")
	plt.plot(x,y_mv,color="blue",linestyle="--",label="Multivariate 1D PDF")
	plt.xlim(range_dist)
	plt.yscale('log')
	plt.ylim(bottom=1e-6)
	plt.legend()
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	plt.figure(1)
	plt.plot(x,z_un,color="black",label="Univariate CDF")
	plt.legend()
	pdf.savefig(bbox_inches='tight')
	plt.close(1)	
	pdf.close()


def test_king_3d(n=10000,r0=0.,rc=1.,rt=10):
	loc = np.array([r0,0.0,0.0])
	scl = np.array([[2.0,0.0,0.0],
					[0.0,2.0,0.0],
					[0.0,0.0,1.0]])

	# ----- Generate samples ---------
	mk = mvking(loc=loc,scale=scl,rt=rt)
	uk = mvking(loc=r0,scale=2.0,rt=rt)
	s_mv = mk.rvs(size=n)
	s_uv = uk.rvs(size=n)

	#-------- Density ---------------
	z = mk.pdf(s_mv)

	pdf = PdfPages(filename="Test_King_3D.pdf")
	plt.figure(0)
	plt.scatter(s_mv[:,0],s_mv[:,1],s=1,c=z,label="3D Samples")
	plt.legend()
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.gca().set_aspect('equal', adjustable='box')
	#-------------- Save fig --------------------------
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	plt.figure(0)
	plt.scatter(s_mv[:,2],s_mv[:,0],s=1,c=z,label="3D Samples")
	plt.legend()

	plt.xlabel("X")
	plt.ylabel("Z")
	plt.gca().set_aspect('equal', adjustable='box')
	#-------------- Save fig --------------------------
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	#-------------- Distance -------------------
	plt.figure(0)
	plt.hist(s_uv,bins=100,density=True,histtype='step',label="Univariate")
	plt.hist(s_mv[:,0],bins=100,density=True,histtype='step',label="X")
	plt.hist(s_mv[:,1],bins=100,density=True,histtype='step',label="Y")
	# plt.hist(sample[:,2],bins=100,density=True,histtype='step',label="Z")
	plt.yscale('log')
	plt.ylim(bottom=1e-5)
	plt.legend()
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	
	pdf.close()
	

if __name__ == "__main__":

	# test_edsd()
    
	# test_ggd()

	# test_king()

	# test_eff_1d()
	# test_eff_2d()
	test_eff_3d()

	# test_king_1d()
	# test_king_3d()

	
