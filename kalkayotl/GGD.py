import sys
#------------------- RandomVariable --------------------
import numpy as np
import scipy.stats as st
from pytensor.tensor.variable import TensorVariable
from pytensor.tensor.random.op import RandomVariable
from typing import List, Tuple
#-------------------------------------------------------

#---------------- Distribution ----------------------------------
import pytensor.tensor as pt
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.shape_utils import rv_size_is_none
#------------------------------------------------------------

class GeneralizedGammaRV(RandomVariable):
	name: str = "GeneralizedGamma"

	# Provide the minimum number of (output) dimensions for this RV
	# Scalars this is a 1D distribution
	ndim_supp: int = 0

	# Provide the number of (input) dimensions for each parameter of the RV
	# a [0], d [0], p [0]
	ndims_params: List[int] = [0, 0, 0]

	# The NumPy/PyTensor dtype for this RV (e.g. `"int32"`, `"int64"`).
	# The standard in the library is `"int64"` for discrete variables
	# and `"floatX"` for continuous variables
	dtype: str = "floatX"

	# A pretty text and LaTeX representation for the RV
	_print_name: Tuple[str, str] = ("GeneralizedGamma", "\\operatorname{GeneralizedGamma}")

	# If you want to add a custom signature and default values for the
	# parameters, do it like this. Otherwise this can be left out.
	def __call__(self, a=1.0, d=1.0, p=1.0, **kwargs) -> TensorVariable:
		return super().__call__(a, d, p, **kwargs)

	# This is the Python code that produces samples.  Its signature will always
	# start with a NumPy `RandomState` object, then the distribution
	# parameters, and, finally, the size.

	@classmethod
	def rng_fn(
		cls,
		rng: np.random.RandomState,
		a: np.ndarray,
		d: np.ndarray,
		p: np.ndarray,
		size: Tuple[int, ...],
	) -> np.ndarray:
		return st.gengamma.rvs(scale=a, a=d/p, c=p, random_state=rng, size=size)

# Create the actual `RandomVariable` `Op`...
gengamma = GeneralizedGammaRV()


# Subclassing `PositiveContinuous` will dispatch a default `log` transformation
class GeneralizedGamma(PositiveContinuous):
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

	# This will be used by the metaclass `DistributionMeta` to dispatch the
	# class `logp` and `logcdf` methods to the `blah` `Op` defined in the last line of the code above.
	rv_op = gengamma

	# dist() is responsible for returning an instance of the rv_op.
	# We pass the standard parametrizations to super().dist
	@classmethod
	def dist(cls, a=None, d=None, p=None, **kwargs):
		a = pt.as_tensor_variable(a)
		d = pt.as_tensor_variable(d)
		p = pt.as_tensor_variable(p)
		# if param2 is not None and alt_param2 is not None:
		#     raise ValueError("Only one of param2 and alt_param2 is allowed.")
		# if alt_param2 is not None:
		#     param2 = 1 / alt_param2
		# param2 = pt.as_tensor_variable(param2)

		# The first value-only argument should be a list of the parameters that
		# the rv_op needs in order to be instantiated
		return super().dist([a,d,p], **kwargs)

	# moment returns a symbolic expression for the stable moment from which to start sampling
	# the variable, given the implicit `rv`, `size` and `param1` ... `paramN`.
	# This is typically a "representative" point such as the the mean or mode.
	def moment(rv, size, a, d, p):
		moment, _ = pt.broadcast_arrays(a,d)
		if not rv_size_is_none(size):
			moment = pt.full(size, moment)
		return moment

	# Logp returns a symbolic expression for the elementwise log-pdf or log-pmf evaluation
	# of the variable given the `value` of the variable and the parameters `param1` ... `paramN`.
	def logp(value, a, d, p):
		fac1 = -pt.log(pt.gamma(d/p))
		fac2 = pt.log(p)
		fac3 = -d * pt.log(a)
		fac4 = pt.log(value)*(d-1.)
		fac5 = -pt.power(value/a, p)
		logp_expression =  fac1 + fac2 + fac3 + fac4 + fac5

		# A switch is often used to enforce the distribution support domain
		bounded_logp_expression = pt.switch(
			pt.gt(value,0),
			logp_expression,
			-np.inf,
		)

		# We use `check_parameters` for parameter validation. After the default expression,
		# multiple comma-separated symbolic conditions can be added.
		# Whenever a bound is invalidated, the returned expression raises an error
		# with the message defined in the optional `msg` keyword argument.
		return check_parameters(
			bounded_logp_expression,
			a > 0, d > 0, p > 0, msg="a>0, d>0, p > 0",
			)

	# logcdf works the same way as logp. For bounded variables, it is expected to return
	# `-inf` for values below the domain start and `0` for values above the domain end.
	def logcdf(value, a, d, p):
		fac1 = -pt.log(pt.gamma(d/p))
		fac2 = pt.log(pt.gammal(d/p,pt.power(value/a, p)))
		logp_expression =  fac1 + fac2

		# A switch is often used to enforce the distribution support domain
		bounded_logp_expression = pt.switch(
			pt.gt(value, 0),
			logp_expression,
			-np.inf,
		)

		# We use `check_parameters` for parameter validation. After the default expression,
		# multiple comma-separated symbolic conditions can be added.
		# Whenever a bound is invalidated, the returned expression raises an error
		# with the message defined in the optional `msg` keyword argument.
		return check_parameters(
			bounded_logp_expression,
			a > 0, d > 0, p > 0, msg="a > 0, d > 0, p > 0",
			)



if __name__ == "__main__":
	import pymc as pm
	from pymc.distributions.distribution import moment

	# print(pm.draw(gengamma(1.,1.,1., size=(10, 2)), random_seed=1))
	

	# pm.blah = pm.Normal in this example
	# print(GeneralizedGamma.dist(a=1,d=1,p=1))

	# Test that the returned blah_op is still working fine
	# print(pm.draw(gengamma(), random_seed=1))
	# array(-1.01397228)

	# Test the moment method
	# print(moment(gengamma()).eval())
	# array(0.)

	# Test the logp method
	print(pm.logp(gengamma(), [-0.5, 1.0,1.0]).eval())
	# array([-1.04393853, -2.04393853])

	# Test the logcdf method
	print(pm.logcdf(gengamma(), [-0.5, 1.0,1.0]).eval())
	# array([-1.17591177, -0.06914345])
	import numpy as np
	import arviz as az
	import matplotlib.pyplot as plt

	a = 2.0
	d = 1.0 
	p = 1.0

	# alpha = p
	# beta  = d -1

	data = st.gengamma.rvs(scale=a, a=d/p, c=p, size=10000)
	with pm.Model():
		scale = pm.Uniform("scale",lower=0.,upper=10.)
		alpha = pm.Uniform("alpha",lower=0.,upper=5.)
		beta = pm.Uniform("beta", lower=-1.,upper=10.)
		# tau = pm.CustomDist("tau",scale, alpha, beta, logp=logp, random=random,observed=data)
		tau = GeneralizedGamma("tau",a=scale, d=beta+1, p=alpha,observed=data)

		

		approx = pm.fit(
					n=int(1e5),
					method="advi",
					progressbar=True,
					)

		posterior = pm.sample(1000,chains=2)

	# #------------- Plot Loss ----------------------------------
	plt.figure()
	plt.plot(approx.hist[-1000:])
	plt.ylabel("Average Loss")
	plt.show()
	plt.figure()
	az.plot_trace(posterior) 
	plt.show()