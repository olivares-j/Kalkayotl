
"""
King distance prior. Inspired by King 1962
"""
__all__ = ['king', 'King']
import numpy as np
import theano.tensor as tt

from pymc3.util import get_variable_name
from pymc3.distributions.dist_math import bound
from pymc3.distributions.continuous import Continuous,assert_negative_support
from pymc3.distributions.distribution import draw_values, generate_samples

from scipy.stats import rv_continuous
from scipy.optimize import root_scalar
import scipy.integrate as integrate

#=============== King generator ===============================
class king_gen(rv_continuous):
	"King distribution"
	def _pdf(self,x,rt):
		u = 1 + rt**2
		cte = 2.*( (rt/u) - (2.*np.arcsinh(rt)/np.sqrt(u)) + np.arctan(rt))

		a = 1./np.sqrt(1.+  x**2)
		b = 1./np.sqrt(1.+ rt**2) 

		res = (1.0/cte)*(a-b)**2

		return np.where(np.abs(x) < rt,res,np.full_like(x,np.nan))

	def _cdf(self,x,rt):
		u   = 1 + rt**2
		cte = 2.*( (rt/u) - (2.*np.arcsinh(rt)/np.sqrt(u)) + np.arctan(rt))

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

	def __init__(self, x=None, *args, **kwargs):
		self.x    = x    = tt.as_tensor_variable(x)

		self.mean = 0.0

		self.rt = tt.sqrt(self.x**(-2) - 1)

		assert_negative_support(x, 'x', 'King')

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
		rt = draw_values([self.rt],point=point,size=size)
		return generate_samples(king.rvs,rt=rt,
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

		cte = 2.*((self.rt*self.x**2) - (2.*self.x*tt.arcsinh(self.rt)) + tt.arctan(self.rt))

		log_d = 2.0*tt.log((1.0+value**2)**(-0.5) - self.x)
		
		return bound(log_d,tt.abs_(value) < self.rt)

	def _repr_latex_(self, name=None, dist=None):
		if dist is None:
			dist = self
		rt    = dist.rt
		name = r'\text{%s}' % name
		return r'${} \sim \text{{King}}(\mathit{{tidal_radius}}={})$'.format(name,get_variable_name(rt))

###################################################### TEST ################################################################################

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def test_numpy(n=100000,r0=100.,rc=2.,rt=20.):
	#----- Generate samples ---------
	s = king.rvs(loc=r0,scale=rc,rt=rt/rc,size=n)
	#------ grid -----
	
	range_dist = (r0-1.5*rt,r0+1.5*rt)
	x = np.linspace(range_dist[0],range_dist[1],1000)
	y = king.pdf(x,loc=r0,scale=rc,rt=rt/rc)
	z = king.cdf(x,loc=r0,scale=rc,rt=rt/rc)
	
	pdf = PdfPages(filename="Test_King_numpy.pdf")
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
	
	
if __name__ == "__main__":
	test_numpy()
	print("numpy version OK")