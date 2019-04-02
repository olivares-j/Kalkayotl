'''
Copyright 2019 Javier Olivares Romero

This file is part of Kalkayotl.

    Kalkayotl is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyAspidistra is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kalkayotl.  If not, see <http://www.gnu.org/licenses/>.
'''
from __future__ import absolute_import, unicode_literals, print_function
import emcee
import numpy as np
import scipy.stats as st

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

file_plot     = "Comparison_of_statistics.pdf"
n_walkers     = 50
df            = 3
n_iterations  = 10000
posterior     = st.chi2(df=df)
p0            = posterior.rvs(size=n_walkers).reshape((-1,1))


def log_posterior(x):
	return posterior.logpdf(x)
		

sampler = emcee.EnsembleSampler(n_walkers,1,log_posterior)

state = sampler.run_mcmc(p0, 100)
sampler.reset()
sampler.run_mcmc(state, n_iterations)
sample  = sampler.get_chain()
logpro  = sampler.get_log_prob()

############## STATISTICS #################################
#-------------- map --------------------------------------
idx_map  = np.unravel_index(logpro.argmax(), logpro.shape)
MAP      = sample[idx_map]

sample   = sample.flatten()
#------------- mode ---------------------------------------
x         = np.linspace(0,5,num=1000)
gkde      = st.gaussian_kde(sample)
mode      = x[np.argmax(gkde(x))]

#------------- mean ----------------------------
mu        = np.mean(sample)

#------------- meadian --------------------
ctr      = np.median(sample)

#------------ median ---------------------------
quantiles = np.quantile(sample,[0.16,0.84])

print("Differences:")
print("Mean: ",mu-df)
print("Mode: ",mode-(df-2))
print("Map: ",MAP[0]-(df-2))
			
pdf = PdfPages(file_plot)
plt.hist(sample, 100, color="k", histtype="step")
plt.axvline(x=MAP,color="black",label="map")
plt.axvline(x=mu,color="orange",label="mean")
plt.axvline(x=ctr,color="green",label="median")
plt.axvline(x=mode,color="blue",label="mode")
plt.axvline(x=quantiles[0],linestyle=":",label="quantiles")
plt.axvline(x=quantiles[1],linestyle=":",label=None)
plt.xlim(0,10)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$p(\theta)$")
plt.legend(loc="best")
pdf.savefig(bbox_inches="tight")
plt.close()
pdf.close()


