import sys
import numpy as np
import scipy.stats as st

def my_mode(sample):
	mins,maxs = np.min(sample),np.max(sample)
	x         = np.linspace(mins,maxs,num=1000)
	try:
		gkde      = st.gaussian_kde(sample.flatten())
		ctr       = x[np.argmax(gkde(x))]
	except:
		ctr = np.nan 
	return ctr

def get_principal(sigma,idx):
	sd_x   = np.sqrt(sigma[idx[0],idx[0]])
	sd_y   = np.sqrt(sigma[idx[1],idx[1]])
	rho_xy = sigma[idx[0],idx[1]]/(sd_x*sd_y)


	# Author: Jake VanderPlas
	# License: BSD
	#----------------------------------------
	level = 1.0 
	sigma_xy2 = rho_xy * sd_x * sd_y

	alpha = 0.5 * np.arctan2(2 * sigma_xy2,(sd_x ** 2 - sd_y ** 2))
	tmp1  = 0.5 * (sd_x ** 2 + sd_y ** 2)
	tmp2  = np.sqrt(0.25 * (sd_x ** 2 - sd_y ** 2) ** 2 + sigma_xy2 ** 2)

	return level*np.sqrt(tmp1 + tmp2), level*np.sqrt(np.abs(tmp1 - tmp2)), alpha* 180. / np.pi


def AngularSeparation(a):

	ra  = np.deg2rad(a[:,0])
	dec = np.deg2rad(a[:,1])

	N   = len(ra)

	A = np.zeros((N,N))

	for i in range(N):
		for j in range(i+1,N):
			A[i,j] = np.degrees(np.arccos(np.sin(dec[i])*np.sin(dec[j]) + np.cos(dec[i])*np.cos(dec[j])*np.cos(ra[i] - ra[j])))

	A = A + A.T

	return A

def CovariancePM(a,case):
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

	elif case == "Vasiliev+2019":
		'''
		Assumes that the covariance is given by Eq. 1 of Vasiliev et al. 2018.
		'''
		result = 0.0008*np.exp(-a/20.0) + 0.004*np.sinc((a/0.5)+0.25)

	elif case == "Lindegren+2020":
		'''
		Assumes that the covariance is given by Eq. 25 of Lindegren et al. 2021 EDR3 astrometric solution.
		'''
		result = 0.000292*np.exp(-a/12.0)

	else:
		sys.exit("Correlation reference not valid!")


	return result

def CovarianceParallax(a,case):
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

	elif case == "Vasiliev+2019":
		'''
		Assumes that the covariance is given by Vasiliev et al. 2018.
		'''
		result = 0.0003*np.exp(-a/20.0) + 0.002*np.sinc((a/0.5)+0.25)

	elif case == "Lindegren+2020":
		'''
		Assumes that the covariance is given by Eq. 24 of Lindegren et al. 2020.
		'''
		result = 0.000142*np.exp(-a/16.0)

	else:
		sys.exit("Correlation reference not valid!")

	return result


if __name__ == "__main__":
	"""
	Test the correlation functions
	"""
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages

	file_plot = "./Angular_correlations.pdf"

	theta = np.linspace(0,6,1000)

	X_l   = CovariancePM(theta,case="Lindegren+2018")
	X_v   = CovariancePM(theta,case="Vasiliev+2019")
	X_e   = CovariancePM(theta,case="Lindegren+2020")

	Y_l   = CovarianceParallax(theta,case="Lindegren+2018")
	Y_v   = CovarianceParallax(theta,case="Vasiliev+2019")
	Y_e   = CovarianceParallax(theta,case="Lindegren+2020")

	pdf = PdfPages(filename=file_plot)
	plt.figure(0)
	plt.suptitle("Covariance of proper motions")
	plt.plot(theta,X_l,label="Lindegren+2018")
	plt.plot(theta,X_v,label="Vasiliev+2019")
	plt.plot(theta,X_e,label="Lindegren+2020")
	plt.xlabel("Angular separation [deg]")
	plt.ylabel("Covariance [(mas/yr)^2]")
	plt.legend()
	pdf.savefig(bbox_inches='tight')
	plt.close(0)

	plt.figure(0)
	plt.suptitle("Covariance of parallax")
	plt.plot(theta,Y_l,label="Lindegren+2018")
	plt.plot(theta,Y_v,label="Vasiliev+2019")
	plt.plot(theta,Y_e,label="Lindegren+2020")
	plt.xlabel("Angular separation [deg]")
	plt.ylabel("Covariance [mas^2]")
	plt.legend()
	pdf.savefig(bbox_inches='tight')
	plt.close(0)
	pdf.close()





