import sys
import numpy as np
import pytensor
from pytensor import tensor as tt

'''
The following transformation have been taken from pygaia (https://github.com/agabrown/PyGaia)
Copyright (c) 2012-2019 Anthony Brown, Gaia Data Processing and Analysis Consortium
'''

#--------------------------- Rotation from Galactic to ICRS --------------------------------
def elementary_rotation_matrix(axis, rotationAngle):
		"""
		Construct an elementary rotation matrix describing a rotation around the x, y, or
		z-axis.
		Parameters
		----------
		axis : str
				Axis around which to rotate ("x", "X", "y", "Y", "z", or "Z")
		rotationAngle : float
				The rotation angle in radians
		Returns
		-------
		rmat : array
				The rotation matrix
		Raises
		------
		ValueError
				If an unsupported rotation axis string is supplied.
		Examples
		--------
		>>> rotmat = elementaryRotationMatrix("y", np.pi/6.0)
		"""
		if axis.upper() == "X":
				return np.array(
						[
								[1.0, 0.0, 0.0],
								[0.0, np.cos(rotationAngle), np.sin(rotationAngle)],
								[0.0, -np.sin(rotationAngle), np.cos(rotationAngle)],
						]
				)
		elif axis.upper() == "Y":
				return np.array(
						[
								[np.cos(rotationAngle), 0.0, -np.sin(rotationAngle)],
								[0.0, 1.0, 0.0],
								[np.sin(rotationAngle), 0.0, np.cos(rotationAngle)],
						]
				)
		elif axis.upper() == "Z":
				return np.array(
						[
								[np.cos(rotationAngle), np.sin(rotationAngle), 0.0],
								[-np.sin(rotationAngle), np.cos(rotationAngle), 0.0],
								[0.0, 0.0, 1.0],
						]
				)
		else:
				raise ValueError("Unknown rotation axis " + axis + "!")


# Astronomical Unit in meter, IAU constant and defining length
_auInMeter = 149597870700.0

# AU expressed in mas*pc or muas*kpc
_auMasParsec = 1000.0

# Number of seconds in Julian year
_julianYearSeconds = 365.25 * 86400.0

# AU expressed in km*yr/s
_auKmYearPerSec = _auInMeter/(_julianYearSeconds*1000.0)

# Galactic pole in ICRS coordinates (see Hipparcos Explanatory Vol 1 section 1.5, and
# Murray, 1983, # section 10.2)
_alphaGalPole = np.deg2rad(192.85948)
_deltaGalPole = np.deg2rad(27.12825)

# The galactic longitude of the ascending node of the galactic plane on the equator of
# ICRS (see Hipparcos Explanatory Vol 1 section 1.5, and Murray, 1983, section 10.2)
_omega = np.deg2rad(32.93192)

# Rotation matrix for the transformation from ICRS to Galactic coordinates. See equation
# (4.25) in chapter 4.5 of "Astrometry for Astrophysics", 2012, van Altena et al.
_matA = elementary_rotation_matrix("z", np.pi / 2.0 + _alphaGalPole)
_matB = elementary_rotation_matrix("x", np.pi / 2.0 - _deltaGalPole)
_matC = elementary_rotation_matrix("z", -_omega)
_rotationMatrixIcrsToGalactic = np.dot(_matC, np.dot(_matB, _matA))

# Alternative way to calculate the rotation matrix from ICRS to Galactic coordinates.
# First calculate the vectors describing the Galactic coordinate reference frame
# expressed within the ICRS.
#
# _vecN = array([0,0,1])
# _vecG3 = array([np.cos(_alphaGalPole)*np.cos(_deltaGalPole),
#       np.sin(_alphaGalPole)*np.cos(_deltaGalPole), np.sin(_deltaGalPole)])
# _vecG0 = np.cross(_vecN,_vecG3)
# _vecG0 = _vecG0/np.sqrt(np.dot(_vecG0,_vecG0))
# _vecG1 = -np.sin(_omega)*np.cross(_vecG3,_vecG0)+np.cos(_omega)*_vecG0
# _vecG2 = np.cross(_vecG3,_vecG1)
# _rotationMatrixIcrsToGalactic=array([_vecG1,_vecG2,_vecG3])

# Rotation matrix for the transformation from Galactic to ICRS coordinates.
_rotationMatrixGalacticToIcrs = np.transpose(_rotationMatrixIcrsToGalactic)

#-----------------------------------------------------------------------------------------------------

################################ TRANSFORMATIONS #####################
def Iden(x):
	return x

def pc2mas(x):
	return 1.e3/x

def mas2pc(x):
	return 1.e3/x

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

#====================================== 3D ==============================================================
def np_radecplx_to_icrs_xyz(a):
	"""
	Convert R.A., Dec. and Parallax to ICRS Cartesian coordinates. The input must be a n x 3 numpy array.
	Note that the angle coordinates follow the astronomical convention of using elevation (declination,
	latitude) rather than its complement (pi/2-elevation), where the latter is commonly used in the
	mathematical treatment of spherical coordinates.

	Parameters
	----------
	n x 3 array with
	ra  - R.A. in degrees
	dec - Dec. in degrees
	plx - Parallax in mas

	Returns
	-------
	n x 3 array with:
	The ICRS Cartesian array of components x, y, z in pc
	"""

	phi   = np.deg2rad(a[:,0])
	theta = np.deg2rad(a[:,1])
	r     = _auMasParsec/a[:,2]

	b   = np.zeros_like(a)

	ctheta = np.cos(theta)
	b[:,0] = r*np.cos(phi)*ctheta
	b[:,1] = r*np.sin(phi)*ctheta
	b[:,2] = r*np.sin(theta)
	
	return b

def np_radecplx_to_galactic_xyz(a):
	"""
	Convert R.A., Dec. and Parallax to Galactic Cartesian coordinates. The input must be a n x 3 numpy array.
	Note that the angle coordinates follow the astronomical convention of using elevation (declination,
	latitude) rather than its complement (pi/2-elevation), where the latter is commonly used in the
	mathematical treatment of spherical coordinates.

	Parameters
	----------
	n x 3 array with
	ra  - R.A. in degrees
	dec - Dec. in degrees
	plx - Parallax in mas

	Returns
	-------
	n x 3 array with:
	The Galactic Cartesian array of components x, y, z in pc
	"""

	icrs_xyz = np_radecplx_to_icrs_xyz(a)

	gala_xyz = np.dot(_rotationMatrixIcrsToGalactic, icrs_xyz.T).T
	
	return gala_xyz

def icrs_xyz_to_radecplx(a):
	"""
	Convert ICRS Cartesian R.A., Dec. and parallax. The input is a 3-dimensional numpy array.
	Note that the angle coordinates follow the astronomical convention of using elevation (declination,
	latitude) rather than its complement (pi/2-elevation), which is commonly used in the mathematical
	treatment of spherical coordinates.
	Parameters
	----------

	x - ICRS Cartesian vector component along the X-axis (pc)
	y - ICRS Cartesian vector component along the Y-axis (pc)
	z - ICRS Cartesian vector component along the Z-axis (pc)
	Returns
	-------

	The spherical coordinates R.A. (deg), Dec. (deg) and parallax (mas)

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
	ra  = tt.rad2deg(phi)   # Degrees
	dec = tt.rad2deg(theta) # Degrees
	plx = _auMasParsec/r    # mas

	#------- Join ------
	radecplx = tt.stack([ra,dec,plx],axis=1)
	return radecplx

def galactic_xyz_to_radecplx(xyz):
	"""
	Convert Galactic XYZ Cartesian to RA, Dec, and Parallax. The input is a 3-dimensional numpy array.
	Note that the angle coordinates follow the astronomical convention of using elevation (declination,
	latitude) rather than its complement (pi/2-elevation), which is commonly used in the mathematical
	treatment of spherical coordinates.
	Parameters
	----------
	x - Galactic Cartesian vector component along the X-axis (pc)
	y - Galactic Cartesian vector component along the Y-axis (pc)
	z - Galactic Cartesian vector component along the Z-axis (pc)

	Returns
	-------
	Array with the spherical coordinates R.A. (degrees), Dec. (degrees) and parallax (mas)

	NOTE THAT THE LONGITUDE ANGLE IS BETWEEN 0 AND +2PI.
	"""
	icrs_xyz = tt.dot(_rotationMatrixGalacticToIcrs, xyz.T).T

	radecplx = icrs_xyz_to_radecplx(icrs_xyz)

	return radecplx
#=================================================================================================


#==================================== 6D =========================================================
def astrometry_and_rv_to_phase_space(X,reference_system="ICRS"):
	if reference_system == "ICRS":
		Y = np_astrometry_and_rv_to_icrs_xyzuvw(X)
	elif reference_system == "Galactic":
		Y = np_astrometry_and_rv_to_galactic_xyzuvw(X)
	else:
		syss.exit("ERROR: unrecognized reference system!")

	return Y

def np_astrometry_and_rv_to_icrs_xyzuvw(X):
	"""
	From the input astrometric plus radial velocity parameters calculate the phase space coordinates. 
	The output phase space coordinates represent ICRS barycentric positions and velocities.
	This function has no mechanism to deal with units. The code is set up such that for input astrometry
	R.A. and Dec. are in degrees, parallaxes and proper motions in mas and mas/yr, and radial velocities in km/s, 
	the phase space coordinates are in pc and km/s. 
	Only positive parallaxes are accepted, an exception is thrown if this condition is not met.
	
	NOTE that the doppler factor k=1/(1-vrad/c) is NOT used in the calculations. This is not a problem for
	sources moving at typical velocities of Galactic stars.

	THIS FUNCTION SHOULD NOT BE USED WHEN THE PARALLAXES HAVE RELATIVE ERRORS LARGER THAN ABOUT 20 PER CENT
	(see http://arxiv.org/abs/1507.02105 for example). For astrometric data with relatively large parallax
	errors you should consider doing your analysis in the data space and use forward modelling of some
	kind.

	Parameters
	----------
	n x 6 array with
	ra       - The R.A. of the source (degrees).
	dec      - The Dec. of the source (degrees).
	parallax - The parallax of the source (in mas)
	murastar - The proper motion in the R.A., multiplied by cos(theta) (mas/yr)
	mudec    - The proper motion in the Dec. (mas/yr)
	vrad     - The radial velocity (km/s)

	Returns
	-------
	n x 6 array with 
	x  - The x component of the ICRS barycentric position vector (in pc).
	y  - The y component of the ICRS barycentric position vector (in pc).
	z  - The z component of the ICRS barycentric position vector (in pc).
	vx - The x component of the ICRS barycentric velocity vector (in km/s).
	vy - The y component of the ICRS barycentric velocity vector (in km/s).
	vz - The z component of the ICRS barycentric velocity vector (in km/s).
	"""
	Y = np.zeros_like(X)

	ra       = X[:,0]
	dec      = X[:,1]
	parallax = X[:,2]
	murastar = X[:,3]
	mudec    = X[:,4]
	vrad     = X[:,5]

	if np.any(parallax<=0.0):
		raise Exception("One or more of the input parallaxes is non-positive")

	Y[:,:3] = np_radecplx_to_icrs_xyz(X[:,:3])
	p, q, r = np_normalTriad(np.deg2rad(ra), np.deg2rad(dec))

	transverseMotionArray = np.array([murastar*_auKmYearPerSec/parallax, mudec*_auKmYearPerSec/parallax,vrad])

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

def np_astrometry_and_rv_to_galactic_xyzuvw(X):
	"""
	From the input astrometric plus radial velocity parameters calculate the 
	phase space coordinates. The output phase space coordinates represent 
	ICRS barycentric positions and velocities. This function has no mechanism to deal with units. 
	The code is set up such that for input astrometry R.A. and Dec. are in degrees, 
	parallaxes and proper motions in mas and mas/yr, and radial velocities in km/s, 
	the phase space coordinates are in pc and km/s. 
	
	Only positive parallaxes are accepted, an exception is thrown if this condition is not met.
	
	NOTE that the doppler factor k=1/(1-vrad/c) is NOT used in the calculations. This is not a problem for
	sources moving at typical velocities of Galactic stars.

	THIS FUNCTION SHOULD NOT BE USED WHEN THE PARALLAXES HAVE RELATIVE ERRORS LARGER THAN ABOUT 20 PER CENT
	(see http://arxiv.org/abs/1507.02105 for example). For astrometric data with relatively large parallax
	errors you should consider doing your analysis in the data space and use forward modelling of some
	kind.

	Parameters
	----------
	n x 6 array with
	ra       - The R.A. of the source (degrees).
	dec      - The Dec. of the source (degrees).
	parallax - The parallax of the source (in mas)
	murastar - The proper motion in the R.A., multiplied by cos(theta) (mas/yr)
	mudec    - The proper motion in the Dec. (mas/yr)
	vrad     - The radial velocity (km/s)

	Returns
	-------
	n x 6 array with 
	x  - The x component of the Galactic barycentric position vector (in pc).
	y  - The y component of the Galactic barycentric position vector (in pc).
	z  - The z component of the Galactic barycentric position vector (in pc).
	vx - The x component of the Galactic barycentric velocity vector (in km/s).
	vy - The y component of the Galactic barycentric velocity vector (in km/s).
	vz - The z component of the Galactic barycentric velocity vector (in km/s).
	"""
	icrs_xyzuvw = np_astrometry_and_rv_to_icrs_xyzuvw(X)
	gala_xyzuvw = np_icrs_xyzuvw_to_galactic_xyzuvw(icrs_xyzuvw)

	return gala_xyzuvw

def np_icrs_xyzuvw_to_galactic_xyzuvw(v):
	"""
	Rotate ICRS Cartesian coordinates to Galactic Cartesian.
	The input is a 6D x n numpy array. 

	Parameters
	----------
	v : 6 x n array in original reference system.

	Returns
	-------
	vrot : 6 x n array after rotation
	"""

	xyz = np.dot(_rotationMatrixIcrsToGalactic, v[:,:3].T).T
	uvw = np.dot(_rotationMatrixIcrsToGalactic, v[:,3:].T).T

	gala = np.hstack([xyz,uvw])
	return gala

def np_galactic_xyzuvw_to_icrs_xyzuvw(v):
	"""
	Rotate Galactic Cartesian coordinates to ICRS Cartesian.
	The input is a 6D x n numpy array. 

	Parameters
	----------
	v : 6 x n array in original reference system.

	Returns
	-------
	vrot : 6 x n array after rotation
	"""

	xyz = np.dot(_rotationMatrixGalacticToIcrs, v[:,:3].T).T
	uvw = np.dot(_rotationMatrixGalacticToIcrs, v[:,3:].T).T

	icrs = np.hstack([xyz,uvw])
	return icrs


def icrs_xyzuvw_to_astrometry_and_rv(a):
	"""
	From the given phase space coordinates calculate the astrometry and radial
	velocity. The phase space coordinates are assumed to represent 
	ICRS barycentric (i.e. centred on the Sun) positions and velocities. 
	This function has no mechanism to deal with units. 
	The velocity units are always assumed to be km/s, and the code is set up 
	such that for positions in pc, the return units for the astrometry are:
	deg, deg, mas, mas/yr and km/s.

	NOTE: The doppler factor k=1/(1-vrad/c) is NOT used in the calculations. 
	This is not a problem for sources moving at typical velocities of Galactic stars.

	Parameters
	----------
	N x 6 array with:
	x -  The x component of the ICRS barycentric position vector (in pc).
	y -  The y component of the ICRS barycentric position vector (in pc).
	z -  The z component of the ICRS barycentric position vector (in pc).
	vx - The x component of the ICRS barycentric velocity vector (in km/s).
	vy - The y component of the ICRS barycentric velocity vector (in km/s).
	vz - The z component of the ICRS barycentric velocity vector (in km/s).

	Returns
	-------
	N x 6 array with:
	ra       - The R.A. of the source (degrees).
	dec      - The Dec. of the source (degrees).
	parallax - The parallax of the source (mas)
	mu_alpha - The proper motion in the R.A., multiplied by cos(theta) (mas/yr)
	mu_delta - The proper motion in the Dec. (mas/yr)
	vrad     - The radial velocity (km/s)
	"""

	x  = a[:,0]
	y  = a[:,1]
	z  = a[:,2]
	vx = a[:,3]
	vy = a[:,4]
	vz = a[:,5]

	b  = icrs_xyz_to_radecplx(a[:,:3])
	
	ra  = b[:,0]
	dec = b[:,1]
	plx = b[:,2]

	p, q, r = normalTriad(tt.deg2rad(ra), tt.deg2rad(dec))

	velocities= a[:,3:]

	murastar = tt.sum(p*velocities,axis=1)*plx/_auKmYearPerSec
	mudec    = tt.sum(q*velocities,axis=1)*plx/_auKmYearPerSec
	vrad     = tt.sum(r*velocities,axis=1)

	#------- Join ----------------------------------------------------------
	as_and_rv = tt.stack([ra, dec, plx, murastar, mudec, vrad],axis=1)
	#-----------------------------------------------------------------------

	return as_and_rv


def galactic_xyzuvw_to_astrometry_and_rv(v):
	"""
	Transform Galactic Cartesian phase space coordinates to 
	astrometry and radial velocity.
	The input is a 6D x n tensor. 

	Parameters
	----------
	v : 6 x n array in Galactic cartesian reference system.

	Returns
	-------
	v : 6 x n arraywith ra (deg), dec (deg), parallax (mas)
	    mu_ra (mas/yr), mu_dec (mas/yr), and radial vel (km/s)
	"""

	xyz = tt.dot(_rotationMatrixGalacticToIcrs, v[:,:3].T).T
	uvw = tt.dot(_rotationMatrixGalacticToIcrs, v[:,3:].T).T

	icrs_xyzuvw = tt.concatenate([xyz,uvw],axis=1)

	as_and_rv = icrs_xyzuvw_to_astrometry_and_rv(icrs_xyzuvw)
	return as_and_rv

###################################################### TEST #############################################

def test_3D(stars):
	from pygaia.astrometry.vectorastrometry import cartesian_to_spherical,spherical_to_cartesian
	from pygaia.astrometry.coordinates import CoordinateTransformation
	from pygaia.astrometry.coordinates import Transformations 

	ICRS2GAL = CoordinateTransformation(Transformations.ICRS2GAL)
	GAL2ICRS = CoordinateTransformation(Transformations.GAL2ICRS)

	print("================== Testing 3D ==========================")

	print("---------- Numpy ra, dec, plx to ICRS XYZ---------------")
	np_icrs_xyz = np_radecplx_to_icrs_xyz(stars[:,:3])
	r,phi,theta = cartesian_to_spherical(np_icrs_xyz[:,0],np_icrs_xyz[:,1],np_icrs_xyz[:,2])
	new_stars = np.column_stack([np.rad2deg(phi),np.rad2deg(theta),1000/r])
	np.testing.assert_allclose(new_stars,stars[:,:3],rtol=1e-5)
	print("                      OK                                ")
	print("--------------------------------------------------------")

	print("---------- Numpy ra, dec, plx to Galactic XYZ-----------")
	np_gala_xyz = np_radecplx_to_galactic_xyz(stars[:,:3])
	x,y,z = GAL2ICRS.transform_cartesian_coordinates(np_gala_xyz[:,0],np_gala_xyz[:,1],np_gala_xyz[:,2])
	r,phi,theta = cartesian_to_spherical(x,y,z)
	new_stars = np.column_stack([np.rad2deg(phi),np.rad2deg(theta),1000/r])
	np.testing.assert_allclose(new_stars,stars[:,:3],rtol=1e-5)
	print("                      OK                                ")
	print("--------------------------------------------------------")

	print("---------- Theano ICRS XYZ to ra, dec, plx--------------")
	icrs_xyz = theano.shared(np_radecplx_to_icrs_xyz(stars[:,:3]))
	radecplx = icrs_xyz_to_radecplx(icrs_xyz)
	new_stars = np.array(radecplx.eval())
	np.testing.assert_allclose(new_stars,stars[:,:3],rtol=1e-5)
	print("                      OK                                ")
	print("--------------------------------------------------------")

	print("---------- Theano Galactic XYZ to ra, dec, plx----------")
	gala_xyz = theano.shared(np_radecplx_to_galactic_xyz(stars[:,:3]))
	radecplx = galactic_xyz_to_radecplx(gala_xyz)
	new_stars = np.array(radecplx.eval())
	np.testing.assert_allclose(new_stars,stars[:,:3],rtol=1e-5)
	print("                      OK                                ")
	print("--------------------------------------------------------")

	print("========================================================")


def test_6D(stars):
	from pygaia.astrometry.vectorastrometry import astrometry_to_phase_space,phase_space_to_astrometry
	from pygaia.astrometry.coordinates import CoordinateTransformation
	from pygaia.astrometry.coordinates import Transformations 

	ICRS2GAL = CoordinateTransformation(Transformations.ICRS2GAL)
	GAL2ICRS = CoordinateTransformation(Transformations.GAL2ICRS)

	print("================== Testing 6D ==========================")

	print("---- Numpy Astrometry and RV to ICRS phase space---------")
	np_icrs_ps = np_astrometry_and_rv_to_icrs_xyzuvw(stars)
	phi,theta,plx,muphi,mutheta,rv = phase_space_to_astrometry(
				np_icrs_ps[:,0],np_icrs_ps[:,1],np_icrs_ps[:,2],
				np_icrs_ps[:,3],np_icrs_ps[:,4],np_icrs_ps[:,5],)
	new_stars = np.column_stack([
						np.rad2deg(phi),np.rad2deg(theta),plx,
						muphi,mutheta,rv])
	np.testing.assert_allclose(new_stars,stars,rtol=1e-5)
	print("                      OK                                ")
	print("--------------------------------------------------------")

	print("--- Numpy Astrometry and RV to Galactic phase space-----")
	np_gala_ps = np_astrometry_and_rv_to_galactic_xyzuvw(stars)
	x,y,z = GAL2ICRS.transform_cartesian_coordinates(
							np_gala_ps[:,0],np_gala_ps[:,1],np_gala_ps[:,2])
	u,v,w = GAL2ICRS.transform_cartesian_coordinates(
							np_gala_ps[:,3],np_gala_ps[:,4],np_gala_ps[:,5])
	phi,theta,plx,muphi,mutheta,rv = phase_space_to_astrometry(
																		x,y,z,u,v,w)
	new_stars = np.column_stack([
						np.rad2deg(phi),np.rad2deg(theta),plx,
						muphi,mutheta,rv])
	np.testing.assert_allclose(new_stars,stars,rtol=1e-5)
	print("                      OK                                ")
	print("--------------------------------------------------------")

	print("--- Numpy ICRS phase space to Galactic phase space -----")
	np_icrs_ps = np_astrometry_and_rv_to_icrs_xyzuvw(stars)
	np_gala_ps = np_icrs_xyzuvw_to_galactic_xyzuvw(np_icrs_ps)
	x,y,z = GAL2ICRS.transform_cartesian_coordinates(
							np_gala_ps[:,0],np_gala_ps[:,1],np_gala_ps[:,2])
	u,v,w = GAL2ICRS.transform_cartesian_coordinates(
							np_gala_ps[:,3],np_gala_ps[:,4],np_gala_ps[:,5])
	phi,theta,plx,muphi,mutheta,rv = phase_space_to_astrometry(
																		x,y,z,u,v,w)
	new_stars = np.column_stack([
						np.rad2deg(phi),np.rad2deg(theta),plx,
						muphi,mutheta,rv])
	np.testing.assert_allclose(new_stars,stars,rtol=1e-5)
	print("                      OK                                ")
	print("--------------------------------------------------------")

	print("--- Theano Galactic phase space to Astrometry and RV ---")
	np_gala_ps = np_astrometry_and_rv_to_galactic_xyzuvw(stars)
	gala_ps = theano.shared(np_gala_ps)
	as_and_rv = galactic_xyzuvw_to_astrometry_and_rv(gala_ps)
	new_stars = np.array(as_and_rv.eval())
	np.testing.assert_allclose(new_stars,stars,rtol=1e-5)
	print("                      OK                                ")
	print("--------------------------------------------------------")


	print("--- Theano ICRS phase space to Astrometry and RV -------")
	np_icrs_ps = np_astrometry_and_rv_to_icrs_xyzuvw(stars)
	icrs_ps = theano.shared(np_icrs_ps)
	as_and_rv = icrs_xyzuvw_to_astrometry_and_rv(icrs_ps)
	new_stars = np.array(as_and_rv.eval())
	np.testing.assert_allclose(new_stars,stars,rtol=1e-5)
	print("                      OK                                ")
	print("--------------------------------------------------------")

	print("========================================================")



def test_Rotation(stars):
	
	import pygaia.astrometry.vectorastrometry as vecast
	from pygaia.astrometry.coordinates import CoordinateTransformation
	from pygaia.astrometry.coordinates import Transformations   

	def astrometry_and_rv_to_galactic_cartesian(ra, de, plx, pmra, pmdec, vr):
			""" 
			From observables in ICRS: 
			- angles in degrees, 
			- plx in mas, 
			- proper motion in mas/yr, 
			- los velocity in km/s;
			returns X,Y,Z (in pc) and U,V,W (in km/s).
			"""

			ICRS2GAL = CoordinateTransformation(Transformations.ICRS2GAL)

			l, b = ICRS2GAL.transform_sky_coordinates(np.deg2rad(ra), np.deg2rad(de))
			mul, mub = ICRS2GAL.transform_proper_motions(np.deg2rad(ra), np.deg2rad(de), pmra, pmdec)
			
			return vecast.astrometry_to_phase_space(l, b, plx, mul, mub, vr)


	gx,gy,gz,gu,gv,gw = astrometry_and_rv_to_galactic_cartesian(
											stars[:,0],stars[:,1],stars[:,2],
											stars[:,3],stars[:,4],stars[:,5])

	gal = np.stack([gx,gy,gz,gu,gv,gw],axis=1)

	icr = np_galactic_xyzuvw_to_icrs_xyzuvw(gal)

	ra,dec,plx,mua,mud,vr = vecast.phase_space_to_astrometry(
							icr[:,0],icr[:,1],icr[:,2],
							icr[:,3],icr[:,4],icr[:,5])

	new_stars = np.array([np.rad2deg(ra),np.rad2deg(dec),plx,mua,mud,vr]).T
	np.testing.assert_allclose(new_stars,stars, rtol=1e-5, atol=0)


if __name__ == "__main__":
	stars = np.array([
		               [68.98016279,16.50930235,48.94,63.45,-188.94,54.398],   # Aldebaran
		               [297.69582730,+08.86832120,194.95,536.23,385.29,-26.60] # Altair
		               ])
	test_Rotation(stars)
	test_3D(stars)
	test_6D(stars)

