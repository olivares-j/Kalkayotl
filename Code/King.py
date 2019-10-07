
"""
King distance prior. Inspired by King 1962
"""

import numpy as np
import theano.tensor as tt

from pymc3.theanof import floatX
from pymc3.util import get_variable_name
from pymc3.distributions.dist_math import bound
from pymc3.distributions.continuous import Continuous,PositiveContinuous,BoundedContinuous,assert_negative_support
from pymc3.distributions.distribution import draw_values, generate_samples

from scipy.stats import rv_continuous
from scipy.optimize import root_scalar
import scipy.integrate as integrate

#=============== EFF generator ===============================
class king_gen(rv_continuous):
	"King distribution"
	""" This probability density function is defined x>0"""
	def _pdf(self,x,r0,rc,rt):
		a    = rt/rc
		b    = rc**2 + rt**2
		norm = (rc/b)*(rc*rt - 2.*rc*np.sqrt(b)*np.arcsinh(a) + np.arctan(a)*b)

		d = np.abs(x-r0)

		f = 1./np.sqrt(1.+(d/rc)**2)
		g = 1./np.sqrt(1.+ a**2) 
		h = (f-g)**2

		i = 0.5*h/norm
		return np.where(d < rt,i,np.zeros_like(x))

	def _cdf(self,x,r0,rc,rt):
		a = rt/rc
		b = rc**2 + rt**2
		c = np.arctan(a)
		d = np.abs(x-r0)
		e = (rc/b)*(rc*rt - 2.*rc*np.sqrt(b)*np.arcsinh(a) + c*b)

		f = d/rc
		g = np.arctan(f)
		h = (rc/b)*(rc*d  - 2.*rc*np.sqrt(b)*np.arcsinh(f) + g*b)
		i = h/e

		j = np.where(x < r0,0.5*(1.0-i),i)
		k = np.where(x > r0,0.5*(1.0+j),j)


		l = np.where(x < (r0-rt),np.zeros_like(x),k)
		m = np.where(x > (r0+rt),np.ones_like(x),l)
		return m
				

	def _rvs(self,r0,rc,rt):
		sz, rndm = self._size, self._random_state
		u = rndm.uniform(0.0,0.999,size=sz)

		v = np.zeros_like(u)
		for i in range(sz[0]):
			# try:
			sol = root_scalar(lambda x : self._cdf(x,r0,rc,rt) - u[i],
					bracket=[r0-rt,r0+rt],
					method='brentq')
			# except Exception as e:
			# 	print(u[i])
			# 	print(self._cdf(r0-rt,r0,rc,rt))
			# 	print(self._cdf(r0+rt,r0,rc,rt))
			# 	raise
			v[i] = sol.root
			sol  = None
		return v

king = king_gen(a=0.0,name='King')
#===============================================================


class King(Continuous):
	R"""
	King 1962 log-likelihood.
	The pdf of this distribution is
	.. math::
	   King(x|r_0,r_c,r_t)=K(0)\cdot
	   \left[ \left[1 + \left(\frac{x-r_0}{r_c}\right)^2\right]^{-\frac{1}{2}}
	   \left[1 + \left(\frac{r_t}{r_c}\right)^2\right]^{-\frac{1}{2}}\right]^2


	.. note::
	   This probability distribution function is defined from r_0-r_t to r_0+r_t
	   Notice that it is already normalized.
	   
	========  ==========================================
	Support   :math:`x \in [r_0-r_t, r_0+r_t]`
	========  ==========================================
	Parameters
	----------
	r0: float
		Location parameter :math:`r_0` (``r_0`` > 0) .

	rc: float
		Scale parameter :math:`r_c` (``r_c`` > 0) .

	rt: float
		Tidal radius parameter :math:`r_t` (``r_t`` > 1) .

	Examples
	--------
	.. code-block:: python
		with pm.Model():
			x = pm.King('x', r0=100,rc=2,rt=20)
	"""

	def __init__(self, r0=None, rc=None, rt=None, *args, **kwargs):

		self.r0    = r0    = tt.as_tensor_variable(r0)
		self.rc    = rc    = tt.as_tensor_variable(rc)
		self.rt    = rt    = tt.as_tensor_variable(rt)

		self.mean = self.r0

		assert_negative_support(rc, 'rc', 'King')
		assert_negative_support(rt, 'rt', 'King')

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
		r0,rc,rt = draw_values([self.r0,self.rc,self.rt], 
									point=point,size=size)
		return generate_samples(king.rvs, r0=r0,rc=rc,rt=rt,
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
		r0     = self.r0
		rc     = self.rc
		rt     = self.rt

		a    = rt/rc
		b    = rc**2 + rt**2
		norm = (rc/b)*(rc*rt - 2.*rc*tt.sqrt(b)*tt.arcsinh(a) + tt.arctan(a)*b)

		f = 1./tt.sqrt(1.+((value-r0)/rc)**2)
		g = 1./tt.sqrt(1.+ a**2) 
		profile = (f-g)**2

		density = 0.5*profile/norm

		log_d= tt.log(density)
		
		return bound(log_d,r0 > 0.,rc > 0.,rt > 0.,rc<rt)

	def _repr_latex_(self, name=None, dist=None):
		if dist is None:
			dist = self
		r0    = dist.r0
		rc    = dist.rc
		rt    = dist.rt
		name = r'\text{%s}' % name
		return r'${} \sim \text{{King}}(\mathit{{loc}}={},\mathit{{scale}}={},\mathit{{tidal_radius}}={})$'.format(name,
					get_variable_name(r0),get_variable_name(rc),get_variable_name(rt))

###################################################### TEST ################################################################################

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def test_numpy(n=10000,r0=300.,rc=3.,rt=20.):
	#----- Generate samples ---------
	s = king.rvs(r0=r0,rc=rc,rt=rt,size=n)
	#------ grid -----
	x = np.linspace(0,2.*r0,1000)
	range_dist = (r0-1.5*rt,r0+1.5*rt)
	y = king.pdf(x,r0=r0,rc=rc,rt=rt)
	z = king.cdf(x,r0=r0,rc=rc,rt=rt)
	
	pdf = PdfPages(filename="Test_King_numpy.pdf")
	plt.figure(0)
	plt.hist(s,bins=100,range=range_dist,density=True,color="grey",label="Samples")
	plt.plot(x,y,color="black",label="PDF")
	plt.xlim(range_dist)
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
	
	
if __name__ == "__main__":
	test_numpy()
	print("numpy version OK")