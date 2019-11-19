
"""
Elson, Fall and Freeman distance prior. Inspired by Ellson et al. 1987
"""
__all__ = ['eff', 'EFF']
import numpy as np
import theano.tensor as tt

from pymc3.util import get_variable_name
from pymc3.distributions.dist_math import bound
from pymc3.distributions.continuous import Continuous
from pymc3.distributions.distribution import draw_values, generate_samples

from scipy.stats import rv_continuous
from scipy.optimize import root_scalar
from scipy.special import gamma as gamma_function
import scipy.integrate as integrate
from scipy.special import hyp2f1

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

	def __init__(self,gamma=None, *args, **kwargs):

		super().__init__(*args, **kwargs)

		self.gamma = gamma = tt.as_tensor_variable(gamma)

		self.mean = 0.0

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
		gamma = draw_values([self.gamma],point=point,size=size)
		return generate_samples(eff.rvs,gamma=gamma,
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

		cte = tt.sqrt(np.pi)*tt.gamma(0.5*(gamma-1.))/tt.gamma(0.5*gamma)

		log_d  = -0.5*gamma*tt.log(1.+ value**2) - tt.log(cte)
		return bound(log_d,gamma > 1.)

	def _repr_latex_(self, name=None, dist=None):
		if dist is None:
			dist = self
		gamma = dist.gamma
		name = r'\text{%s}' % name
		return r'${} \sim \text{{EFF}}(\mathit{{\gamma}}={})$'.format(name,get_variable_name(gamma))

###################################################### TEST ################################################################################

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as st

def test_numpy(n=10000,r0=100.,rc=2.,gamma=2):

	# ----- Generate samples ---------
	s = r0 + rc*eff.rvs(gamma=gamma,size=n)

	#------ grid ----------------------
	range_dist = (r0-20*rc,r0+20*rc)
	x = np.linspace(range_dist[0],range_dist[1],1000)
	y = eff.pdf(x,loc=r0,scale=rc,gamma=gamma)
	z = eff.cdf(x,loc=r0,scale=rc,gamma=gamma)

	pdf = PdfPages(filename="Test_EFF_numpy.pdf")
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
	
	
if __name__ == "__main__":
	test_numpy()
	print("numpy version OK")