import numpy as np


def AngularSeparation(a):

	ra  = a[:,0]
	dec = a[:,1]

	N   = len(ra)

	A = np.zeros((N,N))

	for i in range(N):
		for j in range(i+1,N):
			A[i,j] = np.arccos(np.sin(dec[i])*np.sin(dec[j]) + np.cos(dec[i])*np.cos(dec[j])*np.cos(ra[i] - ra[j]))

	A = A + A.T

	return A

def CovariancePM(a):
	'''
	Assumes that the covariance is given by Eq. 18 of Lindegren et al. 2018.
	microarcsec^2 -> 1e-6 mas^2
	Input in degrees.
	Output in [mas/yr]^2
	'''

	result = (800*1e-6)*np.exp(-a/20.0)

	return result

def CovarianceParallax(a):
	'''
	Assumes that the covariance is given by Eq. 16 of Lindegren et al. 2018.
	microarcsec^2 -> 1e-6 mas^2
	Input in degrees.
	Output in mas^2
	'''

	result = (285*1e-6)*np.exp(-a/14.0)

	return result




