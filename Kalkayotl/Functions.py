import sys
import numpy as np


def AngularSeparation(a):

	ra  = np.radians(a[:,0])
	dec = np.radians(a[:,1])

	N   = len(ra)

	A = np.zeros((N,N))

	for i in range(N):
		for j in range(i+1,N):
			A[i,j] = np.degrees(np.arccos(np.sin(dec[i])*np.sin(dec[j]) + np.cos(dec[i])*np.cos(dec[j])*np.cos(ra[i] - ra[j])))

	A = A + A.T

	return A

def CovariancePM(a,case="Lindegren+2018"):
	'''
	Covariance matrix of the proper motions.
	microarcsec^2 -> 1e-6 mas^2
	Input in degrees.
	Output in [mas/yr]^2
	'''
	if case == "Lindegren+2018":
		'''
		Assumes that the covariance is given by Eq. 18 of Lindegren et al. 2018.
		'''
		result = 0.0008*np.exp(-a/20.0)

	elif case == "Vasiliev+2018":
		'''
		Assumes that the covariance is given by Eq. 1 of Vasiliev et al. 2018.
		'''
		result = 0.0008*np.exp(-a/20.0) + 0.004*np.sinc((a/0.5)+0.25)

	else:
		sys.exit("Correlation reference not valid!")


	return result

def CovarianceParallax(a,case="Lindegren+2018"):
	'''
	Covariance matrix of the parallax
	microarcsec^2 -> 1e-6 mas^2
	Input in degrees.
	Output in mas^2
	'''
	if case == "Lindegren+2018":
		'''
		Assumes that the covariance is given by Eq. 16 of Lindegren et al. 2018.
		'''
		result = 0.000285*np.exp(-a/14.0)

	elif case == "Vasiliev+2018":
		'''
		Assumes that the covariance is given by Vasiliev et al. 2018.
		'''
		result = 0.0003*np.exp(-a/20.0) + 0.002*np.sinc((a/0.5)+0.25)

	else:
		sys.exit("Correlation reference not valid!")

	return result




