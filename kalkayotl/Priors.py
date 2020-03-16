
"""
This file contains the non-standard prior
"""

import numpy as np
import theano.tensor as tt

from pymc3.util import get_variable_name
from pymc3.distributions.dist_math import bound
from pymc3.distributions.continuous import PositiveContinuous,Continuous,assert_negative_support
from pymc3.distributions.distribution import draw_values, generate_samples
from pymc3.distributions.continuous import Uniform

from scipy.stats import rv_continuous
from scipy.optimize import root_scalar
from scipy.special import gamma as gamma_function
import scipy.integrate as integrate
from scipy.special import hyp2f1

##################################### EDSD #######################################################

#=============== EDSD generator ===============================
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
#===============================================================


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

	def _repr_latex_(self, name=None, dist=None):
		if dist is None:
			dist = self
		scale = dist.scale
		name = r'\text{%s}' % name
		return r'${} \sim \text{{EDSD}}(\mathit{{scale}}={})$'.format(name,
																		 get_variable_name(scale))

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
################################################################################################################

##################################### EFF #######################################################
#=============== EFF generator ===============================
class eff_gen(rv_continuous):
	"EFF distribution"
	""" This probability density function is defined for x>0"""
	def _pdf(self,x,gamma):

		cte = np.sqrt(np.pi)*gamma_function(0.5*(gamma-1.))/gamma_function(gamma/2.)
		nx = (1. + x**2)**(-0.5*gamma)
		return nx/cte

	def _cdf(self,x,gamma):
		cte = np.sqrt(np.pi)*gamma_function(0.5*(gamma-1.))/gamma_function(gamma/2.)

		a = hyp2f1(0.5,0.5*gamma,1.5,-x**2)

		return 0.5 + x*(a/cte)
				

	def _rvs(self,gamma):
		#---------------------------------------------
		sz, rndm = self._size, self._random_state
		# Uniform between 0.01 and 0.99. It avoids problems with the
		# numeric integrator
		u = rndm.uniform(0.01,0.99,size=sz) 

		v = np.zeros_like(u)

		for i in range(sz[0]):
			try:
				sol = root_scalar(lambda x : self._cdf(x,gamma) - u[i],
				bracket=[-1000.,1000.],
				method='brentq')
			except Exception as e:
				print(u[i])
				print(self._cdf(-1000.0,gamma))
				print(self._cdf(1000.00,gamma))
				raise
			v[i] = sol.root
			sol  = None
		return v

eff = eff_gen(name='EFF')
#===============================================================


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

	def _repr_latex_(self, name=None, dist=None):
		if dist is None:
			dist = self
		gamma = dist.gamma
		name = r'\text{%s}' % name
		return r'${} \sim \text{{EFF}}(\mathit{{\gamma}}={})$'.format(name,get_variable_name(gamma))

################################################################################################################

################################# KING ################################################################
#=============== King generator ===============================
class king_gen(rv_continuous):
	"King distribution"
	def _pdf(self,x,rt):
		"""
		The tidal radius is in units of the core radius
		"""
		cte = 2.*( rt/(1 + rt**2) - 2.*np.arcsinh(rt)/np.sqrt(1.+ rt**2) + np.arctan(rt))

		a = 1./np.sqrt(1. +  x**2)
		u = 1./np.sqrt(1. + rt**2)

		res = ((a-u)**2)/cte

		return np.where(np.abs(x) < rt,res,np.full_like(x,np.nan))

	def _cdf(self,x,rt):
		u   = 1 + rt**2
		cte = 2.*( rt/(1 + rt**2) - 2.*np.arcsinh(rt)/np.sqrt(1.+ rt**2) + np.arctan(rt))

		val = (rt+x)/u - (2.*(np.arcsinh(rt)+np.arcsinh(x))/np.sqrt(u)) + (np.arctan(rt)+np.arctan(x))

		res = val/cte
		
		return res
				

	def _rvs(self,rt):
		#----------------------------------------
		sz, rndm = self._size, self._random_state
		u = rndm.uniform(0.0,1.0,size=sz) 

		v = np.zeros_like(u)

		for i in range(sz[0]):
			try:
				sol = root_scalar(lambda x : self._cdf(x,rt) - u[i],
				bracket=[-rt,rt],
				method='brentq')
			except Exception as e:
				print(u[i])
				print(self._cdf(-rt,rt))
				print(self._cdf(rt,rt))
				raise
			v[i] = sol.root
			sol  = None
		return v

king = king_gen(name='King')


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

		cte = 2*self.scale*(self.rt/(1+self.rt**2) + tt.arctan(self.rt) - 2.*tt.arcsinh(self.rt)/np.sqrt(1.+self.rt**2))

		log_d = 2.*tt.log(v-u) - tt.log(cte)

		return tt.switch(tt.abs_(r) < self.rt,log_d,-1e20) #avoids inf in advi
		# return bound(log_d,tt.abs_(r) < self.rt)

	def _repr_latex_(self, name=None, dist=None):
		if dist is None:
			dist = self
		rt       = dist.rt
		location = dist.location
		scale    = dist.scale
		name = r'\text{%s}' % name
		return r'${} \sim \text{{King}}(\mathit{{loc}}={},\mathit{{scale}}={},\mathit{{tidal_radius}}={})$'.format(name,
			get_variable_name(location),get_variable_name(scale),get_variable_name(rt))

################################# MvUniform ################################################################
class MvUniform(Continuous):
	R"""
	   
	========  ==========================================
	Support   :math:`x \in [loc-scl,loc+scl]`
	========  ==========================================
	Parameters
	----------
	"""

	def __init__(self,location=None,scale=None, *args, **kwargs):
		self.location   = location  = tt.as_tensor_variable(location)
		self.scale      = scale     = tt.as_tensor_variable(scale)

		assert_negative_support(scale, 'scale', 'MvUniform')

		self.mean = self.location

		super().__init__( *args, **kwargs)


	def random(self, point=None, size=None):
		"""
		Draw random values from MvUniform distribution.
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
		location,scale = draw_values([self.location,self.scale],point=point,size=size)

		sample   = Uniform.dist(lower=-1,upper=1).random(size=self.shape)
		return self.location + self.scale*sample


	def logp(self, value):
		"""
		Calculate log-probability of the MvUniform distribution at specified value.
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

		cte = tt.log(tt.prod(1./(2.*self.scale)))

		cnd = tt.all(tt.abs_(r) <= 1,axis=1)

		# Try to catch those cases outside the range with something like an embudo.

		# return tt.switch(cnd,cte,-1e20)
		return bound(cte,cnd)

	def _repr_latex_(self, name=None, dist=None):
		if dist is None:
			dist = self
		location = dist.location
		scale    = dist.scale
		name = r'\text{%s}' % name
		return r'${} \sim \text{{MvUniform}}(\mathit{{loc}}={},\mathit{{scale}}={}={})$'.format(name,
			get_variable_name(location),get_variable_name(scale))

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

def test_eff(n=10000,r0=100.,rc=2.,gamma=2):

	# ----- Generate samples ---------
	s = r0 + rc*eff.rvs(gamma=gamma,size=n)

	#------ grid ----------------------
	range_dist = (r0-20*rc,r0+20*rc)
	x = np.linspace(range_dist[0],range_dist[1],1000)
	y = eff.pdf(x,loc=r0,scale=rc,gamma=gamma)
	z = eff.cdf(x,loc=r0,scale=rc,gamma=gamma)

	pdf = PdfPages(filename="Test_EFF.pdf")
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
	plt.legend()
	
	#-------------- Save fig --------------------------
	pdf.savefig(bbox_inches='tight')
	plt.close(1)
	
	pdf.close()

def test_king(n=100000,r0=100.,rc=2.,rt=20.):
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

def test_MvUnif(n=10000,loc=[20.,10],scl=[1,3]):
	nx = 20
	ny = 20

	loc = np.array(loc)
	scl = np.array(scl)

	mvunif = MvUniform.dist(location=loc,scale=scl,shape=(n,2))
	t = mvunif.random().eval()

	lower,upper = loc-scl , loc+scl
	x = np.linspace(lower[0]-1,upper[0]+1, nx)
	y = np.linspace(lower[1]-1,upper[1]+1, ny)
	xy = np.empty((nx*ny,2))
	c = 0
	for i in range(nx):
		for j in range(ny):
			xy[c] = x[i],y[j] 
			c += 1

	z = mvunif.logp(xy).eval()
	
	pdf = PdfPages(filename="Test_MvUnif.pdf")
	plt.figure(0)
	plt.scatter(t[:,0],t[:,1],color="grey",label="Samples")
	plt.scatter(xy[:,0],xy[:,1],c=z,label="PDF")
	plt.legend()
	
	#-------------- Save fig --------------------------
	pdf.savefig(bbox_inches='tight')
	plt.close(0)
	
	pdf.close()
	

if __name__ == "__main__":


	# test_edsd()

	# test_eff()

	# test_king()

	test_MvUnif()
