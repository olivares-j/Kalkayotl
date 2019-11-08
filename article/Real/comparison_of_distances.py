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

#============ Directories and data =================
dir_main   = "/home/javier/Repositories/Kalkayotl/"
dir_out    = dir_main  + "Outputs/"
file_plot  = dir_out   + "Plots/Distance_comparison.pdf"

list_of_clusters  = [
{"name":"Pleiades",     "prior_plot":"King","distance":[135.6,0.1,0.1],   "plot":True, "xlims":[126,146],   "ylims":[126,143]},
{"name":"Ruprecht_147", "prior_plot":"King","distance":[305.0,0.5,0.5],   "plot":False},
{"name":"NGC_1647",     "prior_plot":"King","distance":[586.7,1.1,1.0],   "plot":True, "xlims":[510,685],   "ylims":[520,640]},
{"name":"NGC_2264",     "prior_plot":"King","distance":[722.9,3.5,3.5],   "plot":False},
{"name":"NGC_2682",     "prior_plot":"King","distance":[859.1,1.6,1.5],   "plot":True, "xlims":[700,1000],  "ylims":[810,910]},
{"name":"NGC_2244",     "prior_plot":"King","distance":[1548.8,9.2,9.9],  "plot":False},
{"name":"NGC_188",      "prior_plot":"King","distance":[1864.3,5.4,5.3],  "plot":False},
{"name":"IC_1848",      "prior_plot":"King","distance":[2250.9,13.3,13.4],"plot":False},
{"name":"NGC_2420",     "prior_plot":"King","distance":[2552.9,23.6,25],  "plot":False},
{"name":"NGC_6791",     "prior_plot":"King","distance":[4530.8,47.8,42.4],"plot":True, "xlims":[2000,8000],"ylims":[2700,4500]},
{"name":"NGC_3603",     "prior_plot":"King","distance":[9489.4,359,338],  "plot":False}
]

priors = [
# {"name":"EDSD",     "color":"black" ,"marker":"" },
{"name":"Uniform",  "color":"blue"  ,"marker":"v" },
{"name":"Gaussian", "color":"orange","marker":"^" },
# {"name":"Cauchy",   "color":"pink"  ,"marker":"" },
{"name":"GMM",      "color":"green" ,"marker":"<" },
{"name":"EFF",      "color":"purple","marker":">" },
{"name":"King",     "color":"cyan" , "marker":"s" }
]

names = np.array([cluster["name"] for cluster in list_of_clusters],dtype="str")

n_priors   = len(priors)
n_clusters = len(list_of_clusters)
n_clusters_plot = np.sum([cluster["plot"] for cluster in list_of_clusters])

distance_CG = np.array([cluster["distance"] for cluster in list_of_clusters])

distance_KK = np.full((n_clusters,n_priors,3,2),np.nan)


pdf = PdfPages(filename=file_plot)
fig, axes = plt.subplots(n_clusters_plot, 1,num=0,figsize=(6,24))
i = 0
print(30*"=")
print(10*"-"+" Mean uncertainties "+10*"-")
print(" Cluster  | BJ+2018  | Kalkayotl | Difference ")
for j,cluster in enumerate(list_of_clusters):
    dir_chains = dir_out   + "Real/" + cluster["name"] + "/"
    file_data  = dir_main  + "Data/" + cluster["name"] + ".csv"
    #=======================================================================================================================

    for k,prior in enumerate(priors):

        #------- Read cluster parameters ---------------------------------------------------------------------------
        file_csv = dir_chains + prior["name"] +"/Cluster_mode.csv"
        df_pars  = pn.read_csv(file_csv,usecols=["Parameter","mode","upper","lower"])
        pars_loc = list(map(lambda x: ("loc" in x),df_pars["Parameter"])) 
        df_pars  = df_pars.loc[pars_loc]

        distance_KK[j,k,0,:] = df_pars["mode"]
        distance_KK[j,k,1,:] = df_pars["upper"] - df_pars["mode"]
        distance_KK[j,k,2,:] = df_pars["mode"]  - df_pars["lower"]
        

    #================================== Plot points ==========================================================================
    if cluster["plot"]:
        #------ Read distance modes --------------------------------------------------------
        file_csv = dir_chains + cluster["prior_plot"] +"/Sources_mode.csv"
        infered  = pn.read_csv(file_csv,usecols=["ID","mode","lower","upper"])
        infered.sort_values(by="ID",inplace=True)
        infered.set_index("ID",inplace=True)
        #-------- Read data ------------------------------------------------
        data = pn.read_csv(file_data,usecols=["ID","rest","b_rest","B_rest","parallax","parallax_error"])
        data.sort_values(by="ID",inplace=True)
        data.set_index("ID",inplace=True)

        #------------ Merge with data ---------------------------------------------------------
        df         = pn.merge(data,infered,left_index=True, right_index=True,suffixes=("_","_b"))

        df["Frac"]      = df.apply(lambda x: x["parallax_error"]/x["parallax"], axis = 1)
        df.sort_values(by="Frac",inplace=True,ascending=True)

        #---------- Line --------------
        x_pts  = np.linspace(cluster["xlims"][0],cluster["xlims"][1],100)

        x     = df["rest"]
        y     = df["mode"]
        clr   = np.abs(df["Frac"])
        x_err = np.vstack((df["rest"]  -df["b_rest"],
        	               df["B_rest"]-df["rest"]))
        y_err = np.vstack((df["mode"]  -df["lower"],
        	               df["upper"] -df["mode"]))

        mean_x = np.mean(x_err.flatten())
        mean_y = np.mean(y_err.flatten())

        print("{0}  : {1:02.1f} , {2:02.1f}, {3:02.1f}".format(cluster["name"],mean_x,mean_y,100*(mean_x/mean_y)))

        axes[i].errorbar(x,y,yerr=y_err,xerr=x_err,
        	fmt='none',ls='none',marker="o",ms=5,
        	ecolor="grey",elinewidth=0.01,zorder=0,label=None)
        axes[i].plot(x_pts,x_pts,color="black",linestyle='--',linewidth=3,zorder=1)
        points = axes[i].scatter(x,y,s=20,c=clr,marker="s",zorder=2,vmin=0.01,vmax=0.1,
            cmap="viridis")#,norm=mcolors.LogNorm())
        axes[i].annotate(cluster["name"],xy=(0.01,0.9),xycoords="axes fraction")
        axes[i].set_ylabel("Kalkayotl distance [pc]")
        axes[i].set_xlim(cluster["xlims"])
        axes[i].set_ylim(cluster["ylims"])
        i += 1


fig.subplots_adjust(hspace=0.1)
clrb = fig.colorbar(points,orientation="horizontal",pad=0.05,ax=axes)
clrb.set_label("Fractional uncertainty")
plt.xlabel("BJ+2018 distance [pc]")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()


y = np.arange(0,20*n_clusters,20)

plt.figure(figsize=(8,8))
plt.errorbar(np.zeros(n_clusters),y,
        xerr=distance_CG[:,1:].T,
        ls='none',marker="o",ms=5,
        ecolor="grey",elinewidth=0.01,
        color="grey",
        zorder=1,label="CG+2018")
for k,prior in enumerate(priors):
    plt.errorbar(distance_KK[:,k,0,0]-distance_CG[:,0],y + 3*(k+1),
            xerr=distance_KK[:,k,1:,0].T,
            ls='none',marker=prior["marker"],ms=5,
            ecolor="grey",elinewidth=0.01,
            color=prior["color"],
            zorder=2,label=prior["name"])
    plt.errorbar(distance_KK[:,k,0,1]-distance_CG[:,0],y + 3*(k+1),
            xerr=distance_KK[:,k,1:,1].T,
            ls='none',marker=prior["marker"],ms=5,
            ecolor="grey",elinewidth=0.01,
            color=prior["color"],
            zorder=2,label=None)
plt.yticks(y,names)
plt.legend(
    loc = 'lower center', 
    # bbox_to_anchor=(0.3, 0.7,0.6,0.1),
    ncol= len(priors)+1,
    frameon = True,
    fancybox = True,
    columnspacing=1.0,
    # labelspacing=0.01,
    # fontsize = 'small'
    )
plt.ylim(-20,1.1*y.max())
plt.xscale("symlog")
plt.xlabel("Offset [pc]")
pdf.savefig(bbox_inches='tight')

pdf.close()