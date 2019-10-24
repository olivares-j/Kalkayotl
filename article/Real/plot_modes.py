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
    along with PyAspidistra.  If not, see <http://www.gnu.org/licenses/>.
'''
#------------ LOAD LIBRARIES -------------------
from __future__ import absolute_import, unicode_literals, print_function
import sys
import numpy as np
import pandas as pn

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

list_of_clusters  = [
{"name":"Pleiades",     "prior":"King","distance":136.0},
{"name":"Ruprecht_147", "prior":"King","distance":305.0},
{"name":"NGC_3603",     "prior":"King","distance":9493.0},
]
xlims = [[126,146],[270,360],[3000,14000]]
ylims = [[126,143],[280,325],[3000,15000]]
#============ Directories and data =================
dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_out    = dir_main  + "Outputs/"

file_plot  = dir_out   + "Plots/Distance_comparison.pdf"


pdf = PdfPages(filename=file_plot)
fig, axes = plt.subplots(3, 1,num=0,figsize=(6,18))
for i,cluster in enumerate(list_of_clusters):
    dir_chains = dir_out   + cluster["name"] + "/"
    file_data  = dir_main  + "Data/" + cluster["name"] + ".csv"
    #=======================================================================================================================

    #-------- Read data ------------------------------------------------
    data = pn.read_csv(file_data,usecols=["ID","rest","b_rest","B_rest","parallax","parallax_error"])
    data.sort_values(by="ID",inplace=True)
    data.set_index("ID",inplace=True)

    #------ Read modes --------------------------------------------------------
    file_csv = dir_chains + cluster["prior"] +"/Sources_mode.csv"
    infered  = pn.read_csv(file_csv,usecols=["ID","mode","lower","upper"])
    infered.sort_values(by="ID",inplace=True)
    infered.set_index("ID",inplace=True)

    #------------ Merge with data ---------------------------------------------------------
    df         = pn.merge(data,infered,left_index=True, right_index=True,suffixes=("_","_b"))

    df["Frac"]      = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
    df.sort_values(by="Frac",inplace=True,ascending=True)

    #================================== Plot points ==========================================================================

    #---------- Line --------------
    x_pts  = np.linspace(xlims[i][0],xlims[i][1],100)

    x     = df["rest"]
    y     = df["mode"]
    clr   = df["Frac"]
    x_err = np.vstack((df["rest"]  -df["b_rest"],
    	               df["B_rest"]-df["rest"]))
    y_err = np.vstack((df["mode"]  -df["lower"],
    	               df["upper"] -df["mode"]))

    print("Kalkayotl mean uncertainties: {0:2.1f}+/-{1:2.1f}".format(np.mean(y_err.flatten()),np.std(y_err.flatten())))
    print("BJ+2018   mean uncertainties: {0:2.1f}+/-{1:2.1f}".format(np.mean(x_err.flatten()),np.std(x_err.flatten())))

    axes[i].errorbar(x,y,yerr=y_err,xerr=x_err,
    	fmt='none',ls='none',marker="o",ms=5,
    	ecolor="grey",elinewidth=0.01,zorder=0,label=None)
    axes[i].plot(x_pts,x_pts,color="black",linestyle='--',linewidth=3,zorder=1)
    points = axes[i].scatter(x,y,s=20,c=clr,marker="s",zorder=2,vmax=0.5,
        cmap="viridis",norm=mcolors.LogNorm())
    axes[i].annotate(cluster["name"],xy=(0.05,0.9),xycoords="axes fraction")
    axes[i].set_ylabel("Kalkayotl distance [pc]")
    axes[i].set_xlim(xlims[i])
    axes[i].set_ylim(ylims[i])

fig.subplots_adjust(hspace=0.1)
clrb = fig.colorbar(points,orientation="horizontal",pad=0.05,ax=axes)
clrb.set_label("Fractional uncertainty")
plt.xlabel("BJ+2018 distance [pc]")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()

pdf.close()