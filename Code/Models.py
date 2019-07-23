import sys
import numpy as np
import pymc3 as pm
from theano import tensor as tt, printing

################################ TRANSFORMATIONS #####################

def Iden(x):
	return x

def pc2mas(x):
	return 1.e3/x

def cartesianToSpherical3D(a):
	"""
	Convert Cartesian to spherical coordinates. The input can be scalars or 1-dimensional numpy arrays.
	Note that the angle coordinates follow the astronomical convention of using elevation (declination,
	latitude) rather than its complement (pi/2-elevation), which is commonly used in the mathematical
	treatment of spherical coordinates.
	Parameters
	----------

	x - Cartesian vector component along the X-axis
	y - Cartesian vector component along the Y-axis
	z - Cartesian vector component along the Z-axis
	Returns
	-------

	The spherical coordinates r=sqrt(x*x+y*y+z*z), longitude phi, latitude theta.

	NOTE THAT THE LONGITUDE ANGLE IS BETWEEN 0 AND +2PI. FOR r=0 AN EXCEPTION IS RAISED.
	"""
	x = a[:,0]
	y = a[:,1]
	z = a[:,2]
	rCylSq=x*x+y*y
	r=tt.sqrt(rCylSq+z*z)
	# if np.any(r==0.0):
	#   raise Exception("Error: one or more of the points is at distance zero.")
	phi = tt.arctan2(y,x)
	phi = tt.where(phi<0.0, phi+2*np.pi, phi)
	theta = tt.arctan2(z,tt.sqrt(rCylSq))
	#-------- Units----------
	phi   = tt.rad2deg(phi)   # Degrees
	theta = tt.rad2deg(theta) # Degrees
	plx   = 1000.0/r           # mas
	#------- Join ------
	res = tt.stack([phi, theta ,plx],axis=1)
	return res

def cartesianToSpherical5D(a):
	"""
	Convert Cartesian to spherical coordinates. The input must be theano tensors.
	Note that the angle coordinates follow the astronomical convention of using elevation (declination,
	latitude) rather than its complement (pi/2-elevation), which is commonly used in the mathematical
	treatment of spherical coordinates.
	Parameters
	----------

	x - Cartesian vector component along the X-axis
	y - Cartesian vector component along the Y-axis
	z - Cartesian vector component along the Z-axis
	vx - Cartesian vector component of velocity along the Phi   axis
	vy - Cartesian vector component of velocity along the Theta axis

	Returns
	-------

	The spherical coordinates:
	longitude phi, 
	latitude theta,
	parallax,
	proper motion phi,
	proper motion theta.

	NOTE THAT THE LONGITUDE ANGLE IS BETWEEN 0 AND +2PI. FOR r=0 AN EXCEPTION IS RAISED.
	"""
	x  = a[:,0]
	y  = a[:,1]
	z  = a[:,2]
	vx = a[:,3]
	vy = a[:,4]

	rCylSq=x*x+y*y
	r=tt.sqrt(rCylSq+z*z)
	phi = tt.arctan2(y,x)
	phi = tt.where(phi<0.0, phi+2*np.pi, phi)
	theta = tt.arctan2(z,tt.sqrt(rCylSq))

	#------- Velocity ------------------------------------
	mu_phi   = 1000.0*vx/(4.74*r) # Proper motion in mas/yr
	mu_theta = 1000.0*vy/(4.74*r) # Proper motion in mas/yr

	#-------- Units----------
	phi   = tt.rad2deg(phi)   # Degrees
	theta = tt.rad2deg(theta) # Degrees
	plx   = 1000.0/r          # mas


	#------- Join ------
	res = tt.stack([phi, theta ,plx, mu_phi, mu_theta],axis=1)
	return res


def normalTriad(phi, theta):
	"""
	Calculate the so-called normal triad [p, q, r] which is associated with a spherical coordinate system .
	The three vectors are:
	p - The unit tangent vector in the direction of increasing longitudinal angle phi.
	q - The unit tangent vector in the direction of increasing latitudinal angle theta.
	r - The unit vector toward the point (phi, theta).
	Parameters
	----------
	phi   - longitude-like angle (e.g., right ascension, ecliptic longitude) in radians
	theta - latitide-like angle (e.g., declination, ecliptic latitude) in radians

	Returns
	-------
	The normal triad as the vectors p, q, r
	"""
	sphi   = tt.sin(phi)
	stheta = tt.sin(theta)
	cphi   = tt.cos(phi)
	ctheta = tt.cos(theta)
	p=array([-sphi, cphi, zeros_like(phi)])
	q=array([-stheta*cphi, -stheta*sphi, ctheta])
	r=array([ctheta*cphi, ctheta*sphi, stheta])
	return p, q, r


def phaseSpaceToAstrometry(a):
	"""
	From the given phase space coordinates calculate the astrometric observables, including the radial
	velocity, which here is seen as the sixth astrometric parameter. The phase space coordinates are
	assumed to represent barycentric (i.e. centred on the Sun) positions and velocities.
	This function has no mechanism to deal with units. The velocity units are always assumed to be km/s,
	and the code is set up such that for positions in pc, the return units for the astrometry are radians,
	milliarcsec, milliarcsec/year and km/s. For positions in kpc the return units are: radians,
	microarcsec, microarcsec/year, and km/s.
	NOTE that the doppler factor k=1/(1-vrad/c) is NOT used in the calculations. This is not a problem for
	sources moving at typical velocities of Galactic stars.
	Parameters
	----------
	x - The x component of the barycentric position vector (in pc or kpc).
	y - The y component of the barycentric position vector (in pc or kpc).
	z - The z component of the barycentric position vector (in pc or kpc).
	vx - The x component of the barycentric velocity vector (in km/s).
	vy - The y component of the barycentric velocity vector (in km/s).
	vz - The z component of the barycentric velocity vector (in km/s).
	Returns
	-------
	phi       - The longitude-like angle of the position of the source (radians).
	theta     - The latitude-like angle of the position of the source (radians).
	parallax  - The parallax of the source (in mas or muas, see above)
	muphistar - The proper motion in the longitude-like angle, multiplied by cos(theta) (mas/yr or muas/yr,
	see above)
	mutheta   - The proper motion in the latitude-like angle (mas/yr or muas/yr, see above)
	vrad      - The radial velocity (km/s)
	"""
	x  = a[:,0]
	y  = a[:,1]
	z  = a[:,2]
	vx = a[:,3]
	vy = a[:,4]
	vz = a[:,5]

	b     = cartesianToSpherical3D(a[:,:3])
	u     = b[:,0]
	phi   = b[:,1]
	theta = b[:,2]

	parallax = 1000.0/u

	p, q, r = normalTriad(phi, theta)

	velocitiesArray= a[:,3:]


	muphistar=zeros_like(parallax)
	mutheta=zeros_like(parallax)
	vrad=zeros_like(parallax)

	for i in range(parallax.size):
		muphistar[i]=dot(p[:,i],velocitiesArray[:,i])*parallax[i]/_auKmYearPerSec
		mutheta[i]=dot(q[:,i],velocitiesArray[:,i])*parallax[i]/_auKmYearPerSec
		vrad[i]=dot(r[:,i],velocitiesArray[:,i])

	return phi, theta, parallax, muphistar, mutheta, vrad

#################################### 1D Model ######################################################


class Model1D(pm.Model):
	'''
	Model to infer the distance of a series of stars
	'''
	def __init__(self,mu_data,sg_data,
		prior="Gaussian",
		parameters={"location":None,"scale": None},
		hyper_alpha=[[0,10]],
		hyper_beta=[0.5],
		hyper_gamma=None,
		transformation="mas",
		name='flavour_1d', model=None):
		# 2) call super's init first, passing model and name
		# to it name will be prefix for all variables here if
		# no name specified for model there will be no prefix
		super().__init__(name, model)
		# now you are in the context of instance,
		# `modelcontext` will return self you can define
		# variables in several ways note, that all variables
		# will get model's name prefix

		self.lower_bound_sd = 1e-5

		#------------------- Data ------------------------------------------------------
		self.N = len(mu_data)

		if self.N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")

		self.T = np.linalg.inv(sg_data)
		#-------------------------------------------------------------------------------

		#============= Transformations ====================================

		if transformation is "mas":
			Transformation = Iden

		elif transformation is "pc":
			Transformation = pc2mas

		else:
			sys.exit("Transformation is not accepted")
		#==================================================================

		#================ Hyper-parameters =====================================
		if hyper_gamma is None:
			shape = 1
		else:
			shape = len(hyper_gamma)

		#------------------------ Location ----------------------------------
		if parameters["location"] is None:
			pm.Uniform("mu",lower=hyper_alpha[0][0],
							upper=hyper_alpha[0][1],
							shape=shape)

		else:
			self.mu = parameters["location"]

		#------------------------ Scale ---------------------------------------
		if parameters["scale"] is None:
			pm.HalfCauchy("sd",beta=hyper_beta[0],shape=shape)
		else:
			self.sd = parameters["scale"]
		#========================================================================

		#------------------ True values ---------------------------------------------
		if prior is "Gaussian":
			pm.Normal("source",mu=self.mu,sd=self.sd,shape=self.N)

		elif prior is "GMM":
			pm.Dirichlet("weights",a=hyper_gamma)

			pm.NormalMixture("source",w=self.weights,
				mu=self.mu,
				sigma=self.sd,
				comp_shape=1,
				shape=self.N)
		
		else:
			sys.exit("The specified prior is not supported")
		#-----------------------------------------------------------------------------

		#----------------- Transformations ----------------------
		pm.Deterministic('true', Transformation(self.source))

		#----------------------- Likelihood ----------------------------------------
		pm.MvNormal('obs', mu=self.true, tau=self.T,observed=mu_data)
		#------------------------------------------------------------------------------
####################################################################################################

######################################### 3D Model ################################################### 
class Model3D(pm.Model):
	'''
	Model to infer the distance and ra dec position of a cluster
	'''
	def __init__(self,mu_data,sg_data,
		prior="Gaussian",
		parameters={"location":None,"scale":None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		transformation=None,
		name='flavour_3d', model=None):
		# 2) call super's init first, passing model and name
		# to it name will be prefix for all variables here if
		# no name specified for model there will be no prefix
		super().__init__(name, model)
		# now you are in the context of instance,
		# `modelcontext` will return self you can define
		# variables in several ways note, that all variables
		# will get model's name prefix

		#------------------- Data ------------------------------------------------------
		N = int(len(mu_data)/3)


		if N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")

		T = np.linalg.inv(sg_data)
		#-------------------------------------------------------------------------------

		#============= Transformations ====================================

		if transformation is "mas":
			Transformation = Iden

		elif transformation is "pc":
			Transformation = cartesianToSpherical3D

		else:
			sys.exit("Transformation is not accepted")
		#==================================================================

		#================ Hyper-parameters =====================================
		if hyper_gamma is None:
			shape = 1
		else:
			shape = len(hyper_gamma)

		#------------------------ Location ----------------------------------
		if parameters["location"] is None:
			pm.Uniform("location_0",lower=hyper_alpha[0][0],
									upper=hyper_alpha[0][1],
									shape=shape)
			pm.Uniform("location_1",lower=hyper_alpha[1][0],
									upper=hyper_alpha[1][1],
									shape=shape)
			pm.Uniform("location_2",lower=hyper_alpha[2][0],
									upper=hyper_alpha[2][1],
									shape=shape)

			#--------- Join variables --------------
			mu = pm.math.concatenate([self.location_0,
								 self.location_1,
								 self.location_2],axis=0)

		else:
			mu = parameters["location"]

		#------------------------ Scale ---------------------------------------
		if parameters["scale"] is None:
			pm.HalfCauchy("sd_0",beta=hyper_beta[0],shape=shape)
			pm.HalfCauchy("sd_1",beta=hyper_beta[1],shape=shape)
			pm.HalfCauchy("sd_2",beta=hyper_beta[2],shape=shape)

			sigma_diag  = pm.math.concatenate([self.sd_0,self.sd_1,self.sd_2],axis=0)
			sigma       = tt.nlinalg.diag(sigma_diag)


			# pm.LKJCorr('chol_corr', eta=hyper_beta[3], n=3)
			# pm.Deterministic('C', tt.fill_diagonal(self.chol_corr[np.zeros((3, 3), dtype=np.int64)], 1.))
			self.C = np.eye(3)

			cov= tt.nlinalg.matrix_dot(sigma, self.C, sigma)
		   

		else:
			cov = parameters["scale"]
		#========================================================================

		#------------------ True values ---------------------------------------------
		if prior is "Gaussian":
			pm.MvNormal("source",mu=mu,cov=cov,shape=(N,3))

		elif prior is "GMM":
			pm.Dirichlet("weights",a=hyper_gamma,shape=shape)

			pm.NormalMixture("distances",w=self.weights,
				mu=self.mu,
				sigma=self.sd,
				comp_shape=1,
				shape=N)
		
		else:
			sys.exit("The specified prior is not supported")
		#-----------------------------------------------------------------------------

		#----------------------- Transformation---------------------------------------
		pm.Deterministic('transformed', Transformation(self.source))
		# transformed_print = tt.printing.Print("transformed")(self.transformed)
		#-----------------------------------------------------------------------------

		#------------ Flatten --------------------------------------------------------
		pm.Deterministic('true', pm.math.flatten(self.transformed))
		#----------------------------------------------------------------------------

		#----------------------- Likelihood ----------------------------------------
		pm.MvNormal('obs', mu=self.true, tau=T,observed=mu_data)
		#------------------------------------------------------------------------------

##################################################################################################

############################ 5D Model ###########################################################
class Model5D(pm.Model):
	'''
	Model to infer the position,distance and proper motions of a cluster
	'''
	def __init__(self,mu_data,sg_data,
		prior="Gaussian",
		parameters={"location":None,"scale":None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		transformation=None,
		name='flavour_5d', model=None):
		# 2) call super's init first, passing model and name
		# to it name will be prefix for all variables here if
		# no name specified for model there will be no prefix
		super().__init__(name, model)
		# now you are in the context of instance,
		# `modelcontext` will return self you can define
		# variables in several ways note, that all variables
		# will get model's name prefix

		#------------------- Data ------------------------------------------------------
		N = int(len(mu_data)/5)


		if N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")

		T = np.linalg.inv(sg_data)
		#-------------------------------------------------------------------------------

		#============= Transformations ====================================

		if transformation is "mas":
			Transformation = Iden

		elif transformation is "pc":
			Transformation = cartesianToSpherical5D

		else:
			sys.exit("Transformation is not accepted")
		#==================================================================

		#================ Hyper-parameters =====================================
		if hyper_gamma is None:
			shape = 1
		else:
			shape = len(hyper_gamma)

		#------------------------ Location ----------------------------------
		if parameters["location"] is None:
			pm.Uniform("location_0",lower=hyper_alpha[0][0],
									upper=hyper_alpha[0][1],
									shape=shape)
			pm.Uniform("location_1",lower=hyper_alpha[1][0],
									upper=hyper_alpha[1][1],
									shape=shape)
			pm.Uniform("location_2",lower=hyper_alpha[2][0],
									upper=hyper_alpha[2][1],
									shape=shape)
			pm.Uniform("location_3",lower=hyper_alpha[3][0],
									upper=hyper_alpha[3][1],
									shape=shape)
			pm.Uniform("location_4",lower=hyper_alpha[4][0],
									upper=hyper_alpha[4][1],
									shape=shape)

			#--------- Join variables --------------
			mu = pm.math.concatenate([self.location_0,
								 self.location_1,
								 self.location_2,
								 self.location_3,
								 self.location_4],axis=0)

		else:
			mu = parameters["location"]

		#------------------------ Scale ---------------------------------------
		if parameters["scale"] is None:
			pm.HalfCauchy("sd_0",beta=hyper_beta[0],shape=shape)
			pm.HalfCauchy("sd_1",beta=hyper_beta[1],shape=shape)
			pm.HalfCauchy("sd_2",beta=hyper_beta[2],shape=shape)
			pm.HalfCauchy("sd_3",beta=hyper_beta[3],shape=shape)
			pm.HalfCauchy("sd_4",beta=hyper_beta[4],shape=shape)

			sigma_diag  = pm.math.concatenate([self.sd_0,self.sd_1,self.sd_2,self.sd_3,self.sd_4],axis=0)
			sigma       = tt.nlinalg.diag(sigma_diag)


			# pm.LKJCorr('chol_corr', eta=hyper_beta[3], n=3)
			# pm.Deterministic('C', tt.fill_diagonal(self.chol_corr[np.zeros((3, 3), dtype=np.int64)], 1.))
			self.C = np.eye(5)

			cov= tt.nlinalg.matrix_dot(sigma, self.C, sigma)
		   

		else:
			cov = parameters["scale"]
		#========================================================================

		#------------------ True values ---------------------------------------------
		if prior is "Gaussian":
			pm.MvNormal("source",mu=mu,cov=cov,shape=(N,5))

		elif prior is "GMM":
			pm.Dirichlet("weights",a=hyper_gamma,shape=shape)

			pm.NormalMixture("distances",w=self.weights,
				mu=self.mu,
				sigma=self.sd,
				comp_shape=1,
				shape=N)
		
		else:
			sys.exit("The specified prior is not supported")
		#-----------------------------------------------------------------------------

		#----------------------- Transformation---------------------------------------
		pm.Deterministic('transformed', Transformation(self.source))
		# transformed_print = tt.printing.Print("transformed")(self.transformed)
		#-----------------------------------------------------------------------------

		#------------ Flatten --------------------------------------------------------
		pm.Deterministic('true', pm.math.flatten(self.transformed))
		#----------------------------------------------------------------------------

		#----------------------- Likelihood ----------------------------------------
		pm.MvNormal('obs', mu=self.true, tau=T,observed=mu_data)
		#------------------------------------------------------------------------------


############################ 6D Model ###########################################################
class Model6D(pm.Model):
	'''
	Model to infer the position,distance and proper motions of a cluster
	'''
	def __init__(self,mu_data,sg_data,
		prior="Gaussian",
		parameters={"location":None,"scale":None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		transformation=None,
		name='flavour_6d', model=None):
		# 2) call super's init first, passing model and name
		# to it name will be prefix for all variables here if
		# no name specified for model there will be no prefix
		super().__init__(name, model)
		# now you are in the context of instance,
		# `modelcontext` will return self you can define
		# variables in several ways note, that all variables
		# will get model's name prefix

		#------------------- Data ------------------------------------------------------
		N = int(len(mu_data)/6)


		if N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")

		T = np.linalg.inv(sg_data)
		#-------------------------------------------------------------------------------

		#============= Transformations ====================================

		if transformation is "mas":
			Transformation = Iden

		elif transformation is "pc":
			Transformation = cartesianToSpherical

		else:
			sys.exit("Transformation is not accepted")
		#==================================================================

		#================ Hyper-parameters =====================================
		if hyper_gamma is None:
			shape = 1
		else:
			shape = len(hyper_gamma)

		#------------------------ Location ----------------------------------
		if parameters["location"] is None:
			pm.Uniform("location_0",lower=hyper_alpha[0][0],
									upper=hyper_alpha[0][1],
									shape=shape)
			pm.Uniform("location_1",lower=hyper_alpha[1][0],
									upper=hyper_alpha[1][1],
									shape=shape)
			pm.Uniform("location_2",lower=hyper_alpha[2][0],
									upper=hyper_alpha[2][1],
									shape=shape)

			#--------- Join variables --------------
			mu = pm.math.concatenate([self.location_0,
								 self.location_1,
								 self.location_2],axis=0)

		else:
			mu = parameters["location"]

		#------------------------ Scale ---------------------------------------
		if parameters["scale"] is None:
			pm.HalfCauchy("sd_0",beta=hyper_beta[0],shape=shape)
			pm.HalfCauchy("sd_1",beta=hyper_beta[1],shape=shape)
			pm.HalfCauchy("sd_2",beta=hyper_beta[2],shape=shape)

			sigma_diag  = pm.math.concatenate([self.sd_0,self.sd_1,self.sd_2],axis=0)
			sigma       = tt.nlinalg.diag(sigma_diag)


			# pm.LKJCorr('chol_corr', eta=hyper_beta[3], n=3)
			# pm.Deterministic('C', tt.fill_diagonal(self.chol_corr[np.zeros((3, 3), dtype=np.int64)], 1.))
			self.C = np.eye(6)

			cov= tt.nlinalg.matrix_dot(sigma, self.C, sigma)
		   

		else:
			cov = parameters["scale"]
		#========================================================================

		#------------------ True values ---------------------------------------------
		if prior is "Gaussian":
			pm.MvNormal("source",mu=mu,cov=cov,shape=(N,6))

		elif prior is "GMM":
			pm.Dirichlet("weights",a=hyper_gamma,shape=shape)

			pm.NormalMixture("distances",w=self.weights,
				mu=self.mu,
				sigma=self.sd,
				comp_shape=1,
				shape=N)
		
		else:
			sys.exit("The specified prior is not supported")
		#-----------------------------------------------------------------------------

		#----------------------- Transformation---------------------------------------
		pm.Deterministic('transformed', Transformation(self.source))
		# transformed_print = tt.printing.Print("transformed")(self.transformed)
		#-----------------------------------------------------------------------------

		#------------ Flatten --------------------------------------------------------
		pm.Deterministic('true', pm.math.flatten(self.transformed))
		#----------------------------------------------------------------------------

		#----------------------- Likelihood ----------------------------------------
		pm.MvNormal('obs', mu=self.true, tau=T,observed=mu_data)
		#------------------------------------------------------------------------------