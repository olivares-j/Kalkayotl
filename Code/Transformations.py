import numpy as np
import theano
from theano import tensor as tt

'''
The following transformation have been taken from pygaia (https://github.com/agabrown/PyGaia)
Copyright (c) 2012-2019 Anthony Brown, Gaia Data Processing and Analysis Consortium
'''
################################ TRANSFORMATIONS #####################

#-------------------------------------------------------------------------
# Astronomical Unit in meter, IAU constant and defining length
_auInMeter = 149597870700.0

# AU expressed in mas*pc or muas*kpc
_auMasParsec = 1000.0

# Number of seconds in Julian year
_julianYearSeconds = 365.25 * 86400.0

# AU expressed in km*yr/s
_auKmYearPerSec = _auInMeter/(_julianYearSeconds*1000.0)

#---------------------------------------------------------------------

def Iden(x):
	return x

def pc2mas(x):
	return 1.e3/x

def cartesianToSpherical(a):
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

	The spherical coordinates longitude phi (degrees), latitude theta (degrees) and parallax (mas)

	NOTE THAT THE LONGITUDE ANGLE IS BETWEEN 0 AND +2PI. FOR r=0 AN EXCEPTION IS RAISED.
	"""
	x = a[:,0]
	y = a[:,1]
	z = a[:,2]

	rCylSq=x*x+y*y
	r=tt.sqrt(rCylSq+z*z)

	# if np.any(r==0.0):
	#   raise Exception("Error: one or more of the points is at distance zero.")
	phi   = tt.arctan2(y,x)
	phi   = tt.where(phi<0.0, phi+2*np.pi, phi)
	theta = tt.arctan2(z,tt.sqrt(rCylSq))
	#-------- Units----------
	phi   = tt.rad2deg(phi)   # Degrees
	theta = tt.rad2deg(theta) # Degrees
	plx   = _auMasParsec/r    # mas
	#------- Join ------
	res = tt.stack([phi, theta ,plx],axis=1)
	return res

def cartesianToSpherical_plus_mu(a):
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
	plx   = _auMasParsec/r          # mas


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
	zeros  = tt.zeros_like(phi)
	sphi   = tt.sin(phi)
	stheta = tt.sin(theta)
	cphi   = tt.cos(phi)
	ctheta = tt.cos(theta)

	
	q = tt.stack([-stheta*cphi, -stheta*sphi, ctheta],axis=1)
	r = tt.stack([ctheta*cphi, ctheta*sphi, stheta],axis=1)
	p = tt.stack([-sphi, cphi, zeros],axis=1)

	return p, q, r


def phaseSpaceToAstrometry_and_RV(a):
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

	b        = cartesianToSpherical(a[:,:3])
	
	phi      = b[:,0]
	theta    = b[:,1]
	parallax = b[:,2]

	p, q, r = normalTriad(tt.deg2rad(phi), tt.deg2rad(theta))

	velocities= a[:,3:]

	muphistar = tt.sum(p*velocities,axis=1)*parallax/_auKmYearPerSec
	mutheta   = tt.sum(q*velocities,axis=1)*parallax/_auKmYearPerSec
	vrad      = tt.sum(r*velocities,axis=1)

	#------- Join ------
	res = tt.stack([phi, theta, parallax, muphistar, mutheta, vrad],axis=1)
	return res

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
	"""
	x  = a[:,0]
	y  = a[:,1]
	z  = a[:,2]
	vx = a[:,3]
	vy = a[:,4]
	vz = a[:,5]

	b        = cartesianToSpherical(a[:,:3])
	
	phi      = b[:,0]
	theta    = b[:,1]
	parallax = b[:,2]

	p, q, r = normalTriad(tt.deg2rad(phi), tt.deg2rad(theta))

	velocities= a[:,3:]

	muphistar = tt.sum(p*velocities,axis=1)*parallax/_auKmYearPerSec
	mutheta   = tt.sum(q*velocities,axis=1)*parallax/_auKmYearPerSec

	#------- Join ------
	res = tt.stack([phi, theta, parallax, muphistar, mutheta],axis=1)
	return res

############################################### NUMPY ######################################################################

def sphericalToCartesian(a):
  """
  Convert spherical to Cartesian coordinates. The input can be scalars or 1-dimensional numpy arrays.
  Note that the angle coordinates follow the astronomical convention of using elevation (declination,
  latitude) rather than its complement (pi/2-elevation), where the latter is commonly used in the
  mathematical treatment of spherical coordinates.
  Parameters
  ----------
  phi   - longitude-like angle (e.g., right ascension, ecliptic longitude) in degrees
  theta - latitide-like angle (e.g., declination, ecliptic latitude) in degrees
  parallax - in mas
  Returns
  -------
  
  The Cartesian vector components x, y, z
  """

  phi   = np.deg2rad(a[:,0])
  theta = np.deg2rad(a[:,1])
  r     = _auMasParsec/a[:,2]

  b     = np.zeros_like(a)

  ctheta = np.cos(theta)
  b[:,0] = r*np.cos(phi)*ctheta
  b[:,1] = r*np.sin(phi)*ctheta
  b[:,2] = r*np.sin(theta)
  
  return b


def np_normalTriad(phi, theta):
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
  sphi   = np.sin(phi)
  stheta = np.sin(theta)
  cphi   = np.cos(phi)
  ctheta = np.cos(theta)
  p      = np.array([-sphi, cphi, np.zeros_like(phi)])
  q      = np.array([-stheta*cphi, -stheta*sphi, ctheta])
  r      = np.array([ctheta*cphi, ctheta*sphi, stheta])
  return p, q, r


def astrometryToPhaseSpace(X):
	"""
	From the input astrometric parameters calculate the phase space coordinates. The output phase space
	coordinates represent barycentric (i.e. centred on the Sun) positions and velocities.
	This function has no mechanism to deal with units. The code is set up such that for input astrometry
	with parallaxes and proper motions in mas and mas/yr, and radial velocities in km/s, the phase space
	coordinates are in pc and km/s. For input astrometry with parallaxes and proper motions in muas and
	muas/yr, and radial velocities in km/s, the phase space coordinates are in kpc and km/s. Only positive
	parallaxes are accepted, an exception is thrown if this condition is not met.
	NOTE that the doppler factor k=1/(1-vrad/c) is NOT used in the calculations. This is not a problem for
	sources moving at typical velocities of Galactic stars.
	THIS FUNCTION SHOULD NOT BE USED WHEN THE PARALLAXES HAVE RELATIVE ERRORS LARGER THAN ABOUT 20 PER CENT
	(see http://arxiv.org/abs/1507.02105 for example). For astrometric data with relatively large parallax
	errors you should consider doing your analysis in the data space and use forward modelling of some
	kind.
	Parameters
	----------
	phi       - The longitude-like angle of the position of the source (degrees).
	theta     - The latitude-like angle of the position of the source (degrees).
	parallax  - The parallax of the source (in mas or muas, see above)
	muphistar - The proper motion in the longitude-like angle, multiplied by cos(theta) (mas/yr or muas/yr,
	see above)
	mutheta   - The proper motion in the latitude-like angle (mas/yr or muas/yr, see above)
	vrad      - The radial velocity (km/s)
	Returns
	-------
	x - The x component of the barycentric position vector (in pc or kpc).
	y - The y component of the barycentric position vector (in pc or kpc).
	z - The z component of the barycentric position vector (in pc or kpc).
	vx - The x component of the barycentric velocity vector (in km/s).
	vy - The y component of the barycentric velocity vector (in km/s).
	vz - The z component of the barycentric velocity vector (in km/s).
	"""
	Y = np.zeros_like(X)

	phi       = X[:,0]
	theta     = X[:,1]
	parallax  = X[:,2]
	muphistar = X[:,3]
	mutheta   = X[:,4]
	vrad      = X[:,5]

	if np.any(parallax<=0.0):
		raise Exception("One or more of the input parallaxes is non-positive")

	Y[:,:3] = sphericalToCartesian(X[:,:3])
	p, q, r = np_normalTriad(np.deg2rad(phi), np.deg2rad(theta))

	transverseMotionArray = np.array([muphistar*_auKmYearPerSec/parallax, mutheta*_auKmYearPerSec/parallax,
	vrad])

	if np.isscalar(parallax):
		velocityArray=np.dot(np.transpose(np.array([p, q, r])),transverseMotionArray)
		vx = velocityArray[0]
		vy = velocityArray[1]
		vz = velocityArray[2]
	else:
		vx = np.zeros_like(parallax)
		vy = np.zeros_like(parallax)
		vz = np.zeros_like(parallax)
	for i in range(parallax.size):
		velocityArray = np.dot(np.transpose(np.array([p[:,i], q[:,i], r[:,i]])), transverseMotionArray[:,i])
		vx[i] = velocityArray[0]
		vy[i] = velocityArray[1]
		vz[i] = velocityArray[2]

	
	Y[:,3] = vx
	Y[:,4] = vy
	Y[:,5] = vz

	return Y

###################################################### TEST ################################################################################

def test3D():
	a = np.array([[10,10,10],
				  [10,10,10]],dtype="float32")
	A = theano.shared(a)
	B = cartesianToSpherical(A)
	b = np.array(B.eval())
	c = sphericalToCartesian(b)
	assert np.allclose(c[0],a[0],atol=1e-5), "Should be [10,10,10] but is {0}".format(c[0])

def test6D():
	a = np.array([[10,10,10,10,10,10],
				  [10,10,10,10,10,10]],dtype="float32")
	A = theano.shared(a)
	B = phaseSpaceToAstrometry_and_RV(A)
	b = np.array(B.eval())
	c = astrometryToPhaseSpace(b)
	assert np.allclose(c[0],a[0],atol=1e-5), "Should be [10,10,10,10,10,10] but is {0}".format(c[0])


if __name__ == "__main__":
	test3D()
	print("3D coordinates OK")
	test6D()
	print("6D coordinates OK")

