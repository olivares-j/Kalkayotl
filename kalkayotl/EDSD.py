
"""
Exponentially Decreasing Space Density Law (Bailer-Jones 2015)
"""

import numpy as np
import theano.tensor as tt

from pymc3.util import get_variable_name
from pymc3.distributions.dist_math import bound
from pymc3.distributions.continuous import PositiveContinuous,assert_negative_support
from pymc3.distributions.distribution import draw_values, generate_samples

from scipy.stats import rv_continuous
from scipy.optimize import root_scalar

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



###################################################### TEST ################################################################################

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def test_numpy(n=1000,L=100.):
	#----- Generate samples ---------
	s = edsd.rvs(L=L,size=n)


	#------ grid -----
	x = np.linspace(0,10.*L,100)
	y = edsd.pdf(x,L=L)
	z = (x**2/(2.*L**3))*np.exp(-x/L)

	pdf = PdfPages(filename="Test_EDSD_numpy.pdf")
	plt.figure(1)
	plt.hist(s,bins=50,density=True,color="grey",label="Samples")
	plt.plot(x,y,color="black",label="PDF")
	plt.plot(x,z,color="red",linestyle="--",label="True")
	plt.legend()
	
	#-------------- Save fig --------------------------
	pdf.savefig(bbox_inches='tight')
	plt.close(1)
	
	pdf.close()
	

def test_theano():
	return True
	
if __name__ == "__main__":
	test_numpy()
	print("numpy version OK")
	# test_theano()
	# print("theano version OK")
