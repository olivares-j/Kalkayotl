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
	# location [0], scale [0], d [0], p [0]
	ndims_params: List[int] = [0, 0, 0, 0]

	# The NumPy/PyTensor dtype for this RV (e.g. `"int32"`, `"int64"`).
	# The standard in the library is `"int64"` for discrete variables
	# and `"floatX"` for continuous variables
	dtype: str = "floatX"

	# A pretty text and LaTeX representation for the RV
	_print_name: Tuple[str, str] = ("GeneralizedGamma", "\\operatorname{GeneralizedGamma}")

	# If you want to add a custom signature and default values for the
	# parameters, do it like this. Otherwise this can be left out.
	def __call__(self, loc=0.0, scale=1.0, d=4.0, p=3.0, **kwargs) -> TensorVariable:
		return super().__call__(loc,scale, d, p, **kwargs)

	# This is the Python code that produces samples.  Its signature will always
	# start with a NumPy `RandomState` object, then the distribution
	# parameters, and, finally, the size.

	@classmethod
	def rng_fn(
		cls,
		rng: np.random.RandomState,
		loc: np.ndarray,
		scale:np.ndarray,
		d: np.ndarray,
		p: np.ndarray,
		size: Tuple[int, ...],
	) -> np.ndarray:
		sample = st.gengamma.rvs(loc=loc,scale=scale, a=d/p, c=p, random_state=rng, size=size)
		# sample = sample[np.where(sample>0.0)[0]][:size]
		# assert sample.shape[0] == size,"Error in sample size!"
		return sample

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
	def dist(cls, loc=None, scale=None, d=None, p=None, a=None, c=None, **kwargs):
		loc   = pt.as_tensor_variable(loc)
		scale = pt.as_tensor_variable(scale)
		if c is not None and p is not None:
		    raise ValueError("Only one of c or p is allowed.")
		if a is not None and d is not None:
		    raise ValueError("Only one of a or d is allowed.")
		if c is not None:
		    p = c
		if a is not None:
		    d = a / p
		d = pt.as_tensor_variable(d)
		p = pt.as_tensor_variable(p)

		# The first value-only argument should be a list of the parameters that
		# the rv_op needs in order to be instantiated
		return super().dist([loc,scale,d,p], **kwargs)

	# moment returns a symbolic expression for the stable moment from which to start sampling
	# the variable, given the implicit `rv`, `size` and `param1` ... `paramN`.
	# This is typically a "representative" point such as the the mean or mode.
	def moment(rv, size, loc, scale, d, p):
		moment,_ = pt.broadcast_arrays(loc+scale*((d-1)/p)**(1/p),d)
		if not rv_size_is_none(size):
			moment = pt.full(size, moment)
		return moment

	# Logp returns a symbolic expression for the elementwise log-pdf or log-pmf evaluation
	# of the variable given the `value` of the variable and the parameters `param1` ... `paramN`.
	def logp(value, loc, scale, d, p):
		y    = (value-loc)
		fac1 = -pt.log(pt.gamma(d/p))
		fac2 = pt.log(p)
		fac3 = -d * pt.log(scale)
		fac4 = pt.log(y)*(d-1.)
		fac5 = -pt.power(y/scale, p)
		logp_expression =  fac1 + fac2 + fac3 + fac4 + fac5

		# A switch is often used to enforce the distribution support domain
		bounded_logp_expression = pt.switch(
			pt.gt(y,0),
			logp_expression,
			-np.inf,
		)

		# We use `check_parameters` for parameter validation. After the default expression,
		# multiple comma-separated symbolic conditions can be added.
		# Whenever a bound is invalidated, the returned expression raises an error
		# with the message defined in the optional `msg` keyword argument.
		return check_parameters(
			bounded_logp_expression,
			# loc >= 0, scale > 0, d > 0, p > 0, msg="loc >= 0, scale > 0, d>0, p > 0",
			scale > 0, d > 0, p > 0, msg="scale > 0, d>0, p > 0",
			)

	# logcdf works the same way as logp. For bounded variables, it is expected to return
	# `-inf` for values below the domain start and `0` for values above the domain end.
	def logcdf(value, loc, scale, d, p):
		y = (value-loc)
		fac1 = -pt.log(pt.gamma(d/p))
		fac2 = pt.log(pt.gammal(d/p,pt.power(y/scale, p)))
		logp_expression =  fac1 + fac2

		# A switch is often used to enforce the distribution support domain
		bounded_logp_expression = pt.switch(
			pt.gt(y, 0),
			logp_expression,
			-np.inf,
		)

		# We use `check_parameters` for parameter validation. After the default expression,
		# multiple comma-separated symbolic conditions can be added.
		# Whenever a bound is invalidated, the returned expression raises an error
		# with the message defined in the optional `msg` keyword argument.
		return check_parameters(
			bounded_logp_expression,
			# loc >= 0,scale > 0, d > 0, p > 0, msg="loc >= 0, scale > 0, d > 0, p > 0",
			scale > 0, d > 0, p > 0, msg="scale > 0, d > 0, p > 0",
			)



if __name__ == "__main__":
	import numpy as np
	import arviz as az
	import matplotlib.pyplot as plt
	import pymc as pm
	from pymc.distributions.distribution import moment

	# print(pm.draw(gengamma(0.0,1.,4.,3., size=(10, 2)), random_seed=1))
	
	# print(GeneralizedGamma.dist(loc=0.0,scale=1,d=1,p=1))
	# print(pm.draw(gengamma(), random_seed=1))

	# mode = moment(gengamma(loc=loc,scale=scale,d=d,p=p)).eval()
	# print(mode)
	# print(pm.logp(gengamma(), mode).eval())
	# sys.exit()

	# # Test the logp method
	# print(pm.logp(gengamma(), [-0.5, 0.0,1.0,4.0,3.0]).eval())

	# # Test the logcdf method
	# print(pm.logcdf(gengamma(), [-0.5,0.0, 1.0,4.0,3.0]).eval())

	# loc, scale, d, p = 0.0, 1.0, 4.0, 3.0


	# rv = st.gengamma(loc=loc,scale=scale, a=d/p, c=p)
	# pers = [0.001, 0.5, 0.999]

	# vals = rv.ppf(pers)

	# np.allclose(pers, rv.cdf(vals))
	# np.allclose(rv.logpdf(vals),pm.logp(gengamma(loc=loc,scale=scale,d=d,p=p),vals).eval())
	# np.allclose(rv.logcdf(vals),pm.logcdf(gengamma(loc=loc,scale=scale,d=d,p=p),vals).eval())

	def kpf(age):
		return  1/(1.0227121683768*age)


	mu = 40
	sd = 10
	d = 4.
	p = 3.
	
	# data = st.norm(loc=1./(1.0227121683768*mu),scale=0.05).rvs(size=100)
	kappa_mu_true = np.array([0.059,0.031, -0.022])
	kappa_sd_true = np.array([0.005,0.005, 0.02])

	kappa_mu_true = np.array([kpf(mu+2),kpf(mu-2),kpf(mu)])
	kappa_sd_true = np.array([0.005,0.005, 0.02])

	data = st.norm(loc=kappa_mu_true,scale=kappa_sd_true).rvs(size=(100,3))
	# 100 age       25.112  4.827
	# 200           25.116  4.872
	#--------- non-central -------
	# 100 age       24.812  4.844
	# 200 age       25.130  4.647

	parameterization = "central"

	model = pm.Model()
	with model:
		# d = pm.HalfNormal("d",sigma=5)
		# p = pm.HalfNormal("p",sigma=5)
		age = GeneralizedGamma("age",loc=mu-sd,scale=sd, d=d, p=p)
		# age = pm.StudentT("age",mu=mu,sigma=sd,nu=1)
		kappa_sd = pm.Exponential("kappa_sd",scale=0.1)
		kappa_mu = pm.Deterministic("kappa_mu",1./(1.0227121683768*age))
		if parameterization == "central":
			kappa = pm.Normal("kappa",mu=kappa_mu,sigma=kappa_sd,shape=3)
		else:
			offset_kappa = pm.Normal("offset_kappa",mu=0.0,sigma=1.0,shape=3)
			kappa = pm.Deterministic("kappa",kappa_mu + offset_kappa*kappa_sd)
		observed = pm.Normal("observed",mu=kappa,sigma=kappa_sd_true, observed=data)

		
	posterior = pm.sample(2000,chains=2,model=model)
	# posterior.extend(pm.sample_prior_predictive(
	# 						samples=1000,
	# 						model=model))

	

	print(az.summary(posterior,#var_names="age",
						stat_focus = "mean",
						hdi_prob=0.975,
						extend=True))

	# az.plot_trace(posterior,figsize=(10,10)) 
	# plt.show()

	# az.plot_dist_comparison(posterior,var_names="age",figsize=(10,30))
	# plt.show()

	# az.plot_dist_comparison(posterior,var_names="kappa_sd",figsize=(10,30))
	# plt.show()