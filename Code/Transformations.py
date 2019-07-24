import numpy as np
from theano import tensor as tt
################################ TRANSFORMATIONS #####################

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


############################# OBSERVED to PHYSICAL ###############################3
def sphericalToCartesian(a):
	"""
	Convert spherical to Cartesian coordinates. The input can be scalars or 1-dimensional numpy arrays.
	Note that the angle coordinates follow the astronomical convention of using elevation (declination,
	latitude) rather than its complement (pi/2-elevation), where the latter is commonly used in the
	mathematical treatment of spherical coordinates.
	Parameters
	----------
	r     - length of input Cartesian vector.
	phi   - longitude-like angle (e.g., right ascension, ecliptic longitude) in radians
	theta - latitide-like angle (e.g., declination, ecliptic latitude) in radians
	Returns
	-------
	The Cartesian vector components x, y, z
	"""
	#----- Units -------
	#Input:Deg,Deg,mas.
	phi   = a[:,0]
	theta = a[:,1]
	plx   = a[:,2]
	#----- Conversion ------
	phi   = np.deg2rad(phi)
	theta = np.deg2rad(theta)
	r     = 1000.0/plx
	ctheta=np.cos(theta)
	x=r*np.cos(phi)*ctheta
	y=r*np.sin(phi)*ctheta
	z=r*np.sin(theta)
	res = np.vstack([x,y,z]).T
	return res

