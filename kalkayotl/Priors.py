
"""
This file contains the non-standard prior
"""

import numpy as np
from pytensor import tensor as tt, function,printing,pp

from pymc3.distributions.dist_math import bound
from pymc3.distributions.continuous import PositiveContinuous,Continuous,assert_negative_support
from pymc3.distributions.distribution import draw_values, generate_samples
from pymc3.distributions.multivariate import _QuadFormBase
from pymc3.theanof import floatX

from .distributions import edsd #,eff,kingmvking
from .EFF import mveff

#====================== 1D ===============================================================
class EDSD(PositiveContinuous):
	R"""
	Exponentially decreasing space density log-likelihood.
	The pdf of this distribution is
	.. math::
	   EDSD(x \mid L) =
		   \frac{x^2}{2L^3}
		   \exp\left(\frac{-x}{L}\right)

	.. note::
	   The parameter ``L`` refers to the scale length of the exponential decay.
	   
	========  ==========================================
	Support   :math:`x \in [0, \infty)`
	========  ==========================================
	Parameters
	----------
	L : float
		Scale parameter :math:`L` (``L`` > 0) .

	Examples
	--------
	.. code-block:: python
		with pm.Model():
			x = pm.EDSD('x', scale=1000)
	"""

	def __init__(self, scale=None, *args, **kwargs):

		super().__init__(*args, **kwargs)

		self.scale = scale = tt.as_tensor_variable(scale)

		self.mean = 3. * self.scale
		# self.variance = (1. - 2 / np.pi) / self.tau

		assert_negative_support(scale, 'scale', 'EDSD')

	def random(self, point=None, size=None):
		"""
		Draw random values from HalfNormal distribution.
		Parameters
		----------
		point : dict, optional
			Dict of variable values on which random values are to be
			conditioned (uses default point if not specified).
		size : int, optional
			Desired size of random sample (returns one sample if not
			specified).
		Returns
		-------
		array
		"""
		scale = draw_values([self.scale], point=point)[0]
		return generate_samples(edsd.rvs, L=scale,
								dist_shape=self.shape,
								size=size)

	def logp(self, value):
		"""
		Calculate log-probability of EDSD distribution at specified value.
		Parameters
		----------
		value : numeric
			Value(s) for which log-probability is calculated. If the log probabilities for multiple
			values are desired the values must be provided in a numpy array or theano tensor
		Returns
		-------
		TensorVariable
		"""
		scale  = self.scale
		log_d  = 2.0 * tt.log(value) - tt.log(2.0 * scale**3) -  value/scale
		return bound(log_d,value >= 0,scale > 0)


	def logcdf(self, value):
		"""
		Compute the log of the cumulative distribution function for EDSD distribution
		at the specified value.
		Parameters
		----------
		value: numeric
			Value(s) for which log CDF is calculated. If the log CDF for multiple
			values are desired the values must be provided in a numpy array or theano tensor.
		Returns
		-------
		TensorVariable
		"""
		scale  = self.scale
		result = 1.0 - tt.exp(-value/scale)*(value**2 + 2. * value * scale + 2. * scale**2)/(2.*scale**2)
		return result
    
class GGD(PositiveContinuous):
	R"""
	Generalized Gamma Distribution, PDF looks like
	.. math::
	   GGD(x \mid L, \alpha, \beta) =
                   \frac{1}{\Gamma(\frac{\beta+1}{\alpha})}
                   \frac{\alpha}{L^{\beta+1}}
		   x^\beta}
		   \exp\left(-(\frac{x}{L})^\beta\right)

	.. note::
	   See Bailer-Jones et al. (2021) for details.
	   
	========  ==========================================
	Support   :math:`x \in [0, \infty)`
	========  ==========================================
	Parameters
	----------
	L : float
		Scale parameter :math:`L` (``L`` > 0) .
	alpha : float
		Additional scale parameter, alpha > 0
	beta : float
		Additional scale parameter, beta > -1. The EDSD is a special case of GDD with alpha=1.0, beta=2.0

	Examples
	--------
	.. code-block:: python
		with pm.Model():
			x = pm.GGD('x', scale=1000, alpha=1.0, beta=2.0)
	"""
    
	def __init__(self, scale=None, alpha=None, beta=None, *args, **kwargs):

		super().__init__(*args, **kwargs)

		self.scale = scale = tt.as_tensor_variable(scale)
		self.alpha = alpha = tt.as_tensor_variable(alpha)
		self.beta = beta = tt.as_tensor_variable(beta)
		self.mean = self.scale * tt.gamma((beta+2.0)/alpha) / tt.gamma((beta+1.0)/alpha)
        
	def random(self, point=None, size=None):
		"""
		Draw random values from HalfNormal distribution.
		Parameters
		----------
		point : dict, optional
			Dict of variable values on which random values are to be
			conditioned (uses default point if not specified).
		size : int, optional
			Desired size of random sample (returns one sample if not
			specified).
		Returns
		-------
		array
		"""
		scale, alpha, beta = draw_values([self.scale, self.alpha, self.beta], point=point)[0]
		return generate_samples(ggd.rvs, L=scale, alpha=alpha, beta=beta,
								dist_shape=self.shape,
								size=size)

	def logp(self, value):
		"""
		Calculate log-probability of GDD distribution at specified value.
		Parameters
		----------
		value : numeric
			Value(s) for which log-probability is calculated. If the log probabilities for multiple
			values are desired the values must be provided in a numpy array or theano tensor
		Returns
		-------
		TensorVariable
		"""
		L  = self.scale
		alpha = self.alpha
		beta = self.beta
		fac1 = -tt.log(tt.gamma((beta+1.0)/alpha))
		fac2 = tt.log(alpha)
		fac3 = -(beta+1.0)*tt.log(L)
		fac4 = beta*tt.log(value)
		fac5 = -tt.power(value/L, alpha)
		log_d =  fac1 + fac2 + fac3 + fac4 + fac5
		return log_d

	def logcdf(self, value):
		"""
		Compute the log of the cumulative distribution function for GGD distribution
		at the specified value.
		Parameters
		----------
		value: numeric
			Value(s) for which log CDF is calculated. If the log CDF for multiple
			values are desired the values must be provided in a numpy array or theano tensor.
		Returns
		-------
		TensorVariable
		"""
		scale  = self.scale
		alpha = self.alpha
		beta = self.beta
		result = tt.log(tt.math.gammal((beta+1.0)/alpha,tt.pow(r/L,alpha)))
		return result
    # gammal should be the lower incomplete gamma function


class EFF(Continuous):
	R"""
	Elson, Fall and Freeman log-likelihood.
	The pdf of this distribution is
	.. math::
	   EFF(x|r_0,r_c,\gamma)=\frac{\Gamma(\gamma/2)}
	   {\sqrt{\pi}\cdot \Gamma(\frac{\gamma-1}{2})}
	   \left[ 1 + x^2\right]^{-\frac{\gamma}{2}}

	========  ==========================================
	Support   :math:`x \in [-\infty, \infty)`
	========  ==========================================
	Parameters
	----------

	gamma: float
		Slope parameter :math:`\gamma` (``\gamma`` > 1) .

	Examples
	--------
	.. code-block:: python
		with pm.Model():
			x = pm.EFF('x',gamma=2)
	"""

	def __init__(self,location,scale=None,gamma=None, *args, **kwargs):

		super().__init__(*args, **kwargs)

		self.gamma    = tt.as_tensor_variable(gamma)
		self.location = tt.as_tensor_variable(location)
		self.scale    = tt.as_tensor_variable(scale)

		self.mean = self.location

	def random(self, point=None, size=None):
		"""
		Draw random values from HalfNormal distribution.
		Parameters
		----------
		point : dict, optional
			Dict of variable values on which random values are to be
			conditioned (uses default point if not specified).
		size : int, optional
			Desired size of random sample (returns one sample if not
			specified).
		Returns
		-------
		array
		"""
		location,scale,gamma = draw_values([self.location,self.scale,self.gamma],point=point,size=size)
		return generate_samples(eff.rvs,loc=location,scale=scale,gamma=gamma,
								dist_shape=self.shape,
								size=size)

	def logp(self, value):
		"""
		Calculate log-probability of EFF distribution at specified value.
		Parameters
		----------
		value : numeric
			Value(s) for which log-probability is calculated. If the log probabilities for multiple
			values are desired the values must be provided in a numpy array or theano tensor
		Returns
		-------
		TensorVariable
		"""
		gamma  = self.gamma
		x      = (self.location-value)/self.scale

		cte = tt.sqrt(np.pi)*self.scale*tt.gamma(0.5*(gamma-1.))/tt.gamma(0.5*gamma)

		log_d  = -0.5*gamma*tt.log(1.+ x**2) - tt.log(cte)
		return bound(log_d,gamma > 1.)


class King(Continuous):
	R"""
	King 1962 log-likelihood.
	The pdf of this distribution is
	.. math::
	   King(x|r_t)=K(0)\cdot
	   \left[ \left[1 + x^2\right]^{-\frac{1}{2}}
	   \left[1 + r_t\right)^2\right]^{-\frac{1}{2}}\right]^2

	Note: The tidal radius must be in units of core radius
	   
	========  ==========================================
	Support   :math:`x \in [-r_t,+r_t]`
	========  ==========================================
	Parameters
	----------

	rt: float
		Tidal radius parameter :math:`r_t` (``r_t`` > 1) .

	Examples
	--------
	.. code-block:: python
		with pm.Model():
			x = pm.King('x', r0=100,rc=2,rt=20)
	"""

	def __init__(self,location=None,scale=None,rt=None, *args, **kwargs):
		self.location   = location  = tt.as_tensor_variable(location)
		self.scale      = scale     = tt.as_tensor_variable(scale)
		self.rt         = rt        = tt.as_tensor_variable(rt)

		assert_negative_support(scale, 'scale', 'King')
		assert_negative_support(rt,     'rt', 'King')

		self.mean = self.location

		super().__init__( *args, **kwargs)

		

	def random(self, point=None, size=None):
		"""
		Draw random values from King's distribution.
		Parameters
		----------
		point : dict, optional
			Dict of variable values on which random values are to be
			conditioned (uses default point if not specified).
		size : int, optional
			Desired size of random sample (returns one sample if not
			specified).
		Returns
		-------
		array
		"""
		location,scale,rt = draw_values([self.location,self.scale,self.rt],point=point,size=size)
		return generate_samples(king.rvs,loc=location,scale=scale,rt=rt,
								dist_shape=self.shape,
								size=size)

	def logp(self, value):
		"""
		Calculate log-probability of King distribution at specified value.
		Parameters
		----------
		value : numeric
			Value(s) for which log-probability is calculated. If the log probabilities for multiple
			values are desired the values must be provided in a numpy array or theano tensor
		Returns
		-------
		TensorVariable
		"""
		r = (value-self.location)/self.scale
		v = 1.0/tt.sqrt(1.+ r**2)
		u = 1.0/tt.sqrt(1.+self.rt**2)

		cte = 2*self.scale*(self.rt/(1+self.rt**2) + tt.arctan(self.rt) - 2.*tt.arcsinh(self.rt)/tt.sqrt(1.+self.rt**2))

		log_d = 2.*tt.log(v-u) - tt.log(cte)

		return tt.switch(tt.abs_(r) < self.rt,log_d,-1e20) #avoids inf in advi
		# return bound(log_d,tt.abs_(r) < self.rt)


#=============================== 3D ======================================================

class MvEFF(_QuadFormBase):
	R"""
	Multivariate Elson, Fall, and Freeman log-likelihood.
	.. math::
	   f(x \mid \pi, T) =
	========  ==========================
	Support   :math:`x \in \mathbb{R}^k`
	Mean      :math:`\mu`
	Variance  :math:`T^{-1}`
	========  ==========================
	Parameters
	----------
	mu: array
		Vector of means.
	cov: array
		Covariance matrix. Exactly one of cov, tau, or chol is needed.
	tau: array
		Precision matrix. Exactly one of cov, tau, or chol is needed.
	chol: array
		Cholesky decomposition of covariance matrix. Exactly one of cov,
		tau, or chol is needed.
	lower: bool, default=True
		Whether chol is the lower tridiagonal cholesky factor.
	"""
	def __init__(self, location, chol, gamma, *args, **kwargs):
		super().__init__(mu=location, chol=chol, *args, **kwargs)

		self.gamma    = tt.as_tensor_variable(gamma)
		self.location = tt.as_tensor_variable(location)
		self.chol     = tt.as_tensor_variable(chol)

		self.mean = self.location


	def random(self, point=None, size=None):
		"""
		Draw random values from HalfNormal distribution.
		Parameters
		----------
		point : dict, optional
			Dict of variable values on which random values are to be
			conditioned (uses default point if not specified).
		size : int, optional
			Desired size of random sample (returns one sample if not
			specified).
		Returns
		-------
		array
		"""

		location,chol,gamma = draw_values([self.location,self.chol,self.gamma],point=point,size=size)

		if location.shape[-1] != chol.shape[-1]:
				raise ValueError("Shapes for location and scale don't match")

		scale = np.dot(chol,chol.T)

		try:
			dist = mveff(loc=location, scale=scale, gamma=gamma)
		except ValueError:
			size += (location.shape[-1],)
			return np.nan * np.zeros(size)

		return dist.rvs(size)

	def logp(self, value):
		"""
		Calculate log-probability of EFF distribution at specified value.
		Parameters
		----------
		value : numeric
			Value(s) for which log-probability is calculated. If the log probabilities for multiple
			values are desired the values must be provided in a numpy array or theano tensor
		Returns
		-------
		TensorVariable
		"""
		quaddist, logdet, ok = self._quaddist(value)

		a = 1.5*tt.log(np.pi)
		b = tt.log(tt.gamma(0.5*(self.gamma-3.0)))
		c = tt.log(tt.gamma(0.5*self.gamma))
		log_cte = a + b -c
		log_d = - 0.5*self.gamma*tt.log1p(quaddist) - log_cte - 3.*logdet
		
		return bound(log_d,ok,self.gamma > 3.001)

class MvKing(_QuadFormBase):
	R"""
	Multivariate King log-likelihood.
	.. math::
	   f(x \mid \pi, T) =
	========  ==========================
	Support   :math:`x \in \mathbb{R}^k`
	Mean      :math:`\mu`
	Variance  :math:`T^{-1}`
	========  ==========================
	Parameters
	----------
	mu: array
		Vector of means.
	cov: array
		Covariance matrix. Exactly one of cov, tau, or chol is needed.
	tau: array
		Precision matrix. Exactly one of cov, tau, or chol is needed.
	chol: array
		Cholesky decomposition of covariance matrix. Exactly one of cov,
		tau, or chol is needed.
	lower: bool, default=True
		Whether chol is the lower tridiagonal cholesky factor.
	"""
	def __init__(self, location, chol, rt, *args, **kwargs):
		super().__init__(mu=location, chol=chol, *args, **kwargs)

		self.rt       = tt.as_tensor_variable(rt)
		self.location = tt.as_tensor_variable(location)
		self.chol     = tt.as_tensor_variable(chol)

		self.mean = self.location

		assert_negative_support(rt,     'rt', 'King')


	def random(self, point=None, size=None):
		"""
		Draw random values from HalfNormal distribution.
		Parameters
		----------
		point : dict, optional
			Dict of variable values on which random values are to be
			conditioned (uses default point if not specified).
		size : int, optional
			Desired size of random sample (returns one sample if not
			specified).
		Returns
		-------
		array
		"""

		location,chol,rt = draw_values([self.location,self.chol,self.rt],point=point,size=size)

		if location.shape[-1] != chol.shape[-1]:
				raise ValueError("Shapes for location and scale don't match")

		scale = np.dot(chol,chol.T)

		try:
			dist = mvking(loc=location, scale=scale, rt=rt)
		except ValueError:
			size += (location.shape[-1],)
			return np.nan * np.zeros(size)

		return dist.rvs(size)

	def logp(self, value):
		"""
		Calculate log-probability of EFF distribution at specified value.
		Parameters
		----------
		value : numeric
			Value(s) for which log-probability is calculated. If the log probabilities for multiple
			values are desired the values must be provided in a numpy array or theano tensor
		Returns
		-------
		TensorVariable
		"""
		quaddist, logdet, ok = self._quaddist(value)

		a = 1. + self.rt**2
		b = (self.rt**3)/(3.*a)
		c = tt.arcsinh(self.rt)/tt.sqrt(a)
		d = tt.arctan(self.rt)

		cte = 4.*np.pi*(b+c-d)

		u = 1./tt.sqrt(1. + quaddist)
		v = 1./tt.sqrt(a)
		log_d = 2.*tt.log(u-v) - tt.log(cte) - 3.*logdet # Here we introduce the scale rc**3

		# result = tt.switch(tt.sqrt(quaddist) < self.rt,log_d,-1.*tt.sqrt(quaddist)-20.) #avoids inf in advi

		return bound(log_d,quaddist < self.rt**2,ok)

###################################################### TEST ################################################################################
import pandas as pd
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as st

def test_MvEFF(n=1000,mc_file=None):
	if mc_file is not None:
		mc = pd.read_csv(mc_file,sep="\t",header=None,skiprows=1,
					names=["Mass","X","Y","Z","U","V","W"])

		pos = mc[["X","Y","Z"]].to_numpy()

		r_mc = np.sqrt(np.sum(pos**2,axis=1))
		z_mc = eff.logpdf(r_mc,gamma=5.0)

	loc = np.array([0.0,0.0,0.0])
	scl = np.array([[1.0,0.0,0.0],
					[0.0,1.0,0.0],
					[0.0,0.0,1.0]])

	chol = np.linalg.cholesky(scl)

	mveff = MvEFF.dist(location=loc,chol=chol,gamma=5.0)
	samples = mveff.random(size=n)

	r = np.sqrt(np.sum(samples**2,axis=1))

	z = eff.logpdf(r,gamma=5.0)

	
	pdf = PdfPages(filename="Prior_MvEFF.pdf")
	# plt.figure(0)
	# plt.scatter(samples[:,0],samples[:,1],c=z,label="Samples")
	# plt.legend()
	# plt.xlabel("X")
	# plt.ylabel("Y")
	# pdf.savefig(bbox_inches='tight')
	# plt.close(0)

	plt.figure(0)
	plt.hist(r,range=[0,5],bins=20,histtype="step",density=True,label="Samples")
	plt.hist(r_mc,range=[0,5],bins=20,histtype="step",density=True,label="McCluster")
	plt.legend()
	plt.xlabel("R [pc]")
	plt.yscale("log")
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	plt.figure(0)
	plt.scatter(r,z,label="Samples")
	plt.scatter(r_mc,z_mc,label="McCluster")
	plt.legend()
	plt.xlabel("R [pc]")
	plt.ylabel("Log Likelihood")
	pdf.savefig(bbox_inches='tight')
	plt.close(0)
	
	pdf.close()

def test_MvKing(n=10000,rt=20.0):
	loc = np.array([0.,0.0,0.0])
	scl = np.array([[1.0,0.0,0.0],
					[0.0,1.0,0.0],
					[0.0,0.0,1.0]])

	chol = np.linalg.cholesky(scl)

	mvk = MvKing.dist(location=loc,chol=chol,rt=rt)
	# samples = mvk.random(size=n)

	mveff = MvEFF.dist(location=loc,chol=2.*chol,gamma=3.0)
	samples = mveff.random(size=n)
	z = mvk.logp(samples)

	# d,z = mvk.logp(samples)

	print(np.min(z.eval()),np.max(z.eval()))
	
	pdf = PdfPages(filename="Prior_MvKing.pdf")
	# plt.figure(0)
	# plt.scatter(d.eval(),z.eval(),s=1)
	# plt.xlabel("Distance")
	# plt.ylabel("Log p")
	# pdf.savefig(bbox_inches='tight')
	# plt.close(0)

	plt.figure(0)
	plt.scatter(samples[:,0],samples[:,1],s=1)
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.gca().set_aspect('equal', adjustable='box')
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	plt.figure(0)
	plt.scatter(samples[:,0],samples[:,2],s=1)
	plt.xlabel("X")
	plt.ylabel("Z")
	plt.gca().set_aspect('equal', adjustable='box')
	pdf.savefig(bbox_inches='tight')
	plt.close(0)
	
	pdf.close()
	


if __name__ == "__main__":

	test_MvEFF(mc_file="/home/jolivares/Repos/Amasijo/Data/EFF_n1000_r1_g5.txt")

	# test_MvKing()
