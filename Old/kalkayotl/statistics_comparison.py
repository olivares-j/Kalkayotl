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

file_plot     = "Statistics_comparison.pdf"
df            = 3
burnin_it     = 100
posterior     = st.chi2(df=df)



def log_posterior(x):
	return posterior.logpdf(x)
		
#--------------------- Repeat -------------------------------
list_walkers = np.array([10,20,30,40,50])
list_iters   = np.array([2,4,6,8,10])*1000

Nw,Ni = len(list_walkers),len(list_iters)

sts   = np.empty((Nw,Ni,3))
for i,walkers in enumerate(list_walkers):
    for j,iters in enumerate(list_iters):
        p0      = posterior.rvs(size=walkers).reshape((-1,1))
        sampler = emcee.EnsembleSampler(walkers,1,log_posterior)
        state = sampler.run_mcmc(p0, burnin_it)
        sampler.reset()
        sampler.run_mcmc(state,iters)
        sample  = sampler.get_chain()
        logpro  = sampler.get_log_prob()

        ############## STATISTICS #################################
        #-------------- map --------------------------------------
        idx_map    = np.unravel_index(logpro.argmax(), logpro.shape)
        sts[i,j,2] = sample[idx_map][0]

        sample     = sample.flatten()
        #------------- mode ---------------------------------------
        x          = np.linspace(0,5,num=1000)
        gkde       = st.gaussian_kde(sample)
        sts[i,j,1] = x[np.argmax(gkde(x))]

        #------------- mean ----------------------------
        sts[i,j,0] = np.mean(sample)


pdf = PdfPages(file_plot)
plt.hist(sample, 100, color="k", histtype="step")
plt.axvline(x=sts[i,j,2],color="black",label="map")
plt.axvline(x=sts[i,j,0],color="orange",label="mean")
plt.axvline(x=np.median(sample),color="green",label="median")
plt.axvline(x=sts[i,j,1],color="blue",label="mode")
plt.axvline(x= np.quantile(sample,0.16),linestyle=":",label="quantiles")
plt.axvline(x= np.quantile(sample,0.84),linestyle=":",label=None)
plt.xlim(0,10)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$p(\theta)$")
plt.legend(loc="best")
pdf.savefig(bbox_inches="tight")
plt.close()

fbias_mean = np.abs(sts[:,:,0] - df)/df
fbias_mode = np.abs(sts[:,:,1] - (df-2))/(df-2)
fbias_map  = np.abs(sts[:,:,2] - (df-2))/(df-2)


xticks = ['']+['{0:d}'.format(walkers) for walkers in list_walkers]
yticks = ['']+['{0:1.0e}'.format(iters) for iters in list_iters]

fig,ax = plt.subplots()
cax = ax.imshow(fbias_mean,origin="lower")
ax.set_xticklabels(xticks)
ax.set_yticklabels(yticks)
ax.set_xlabel("Walkers")
ax.set_ylabel("Iterations")
plt.colorbar(cax,format='%2.1e')
pdf.savefig(bbox_inches="tight")
plt.close()

fig,ax = plt.subplots()
cax = ax.imshow(fbias_mode,origin="lower")
ax.set_xticklabels(xticks)
ax.set_yticklabels(yticks)
ax.set_xlabel("Walkers")
ax.set_ylabel("Iterations")
plt.colorbar(cax,format='%2.1e')
pdf.savefig(bbox_inches="tight")
plt.close()

fig,ax = plt.subplots()
cax = ax.imshow(fbias_map,origin="lower")
ax.set_xticklabels(xticks)
ax.set_yticklabels(yticks)
ax.set_xlabel("Walkers")
ax.set_ylabel("Iterations")
plt.colorbar(cax,format='%2.1e')
pdf.savefig(bbox_inches="tight")
plt.close()


pdf.close()