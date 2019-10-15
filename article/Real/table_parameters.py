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

list_of_cases = [
# "Pleiades",
# "Ruprecht_147",
"NGC_1647",
"IC_1848",
"NGC_3603",
"NGC_6791",
]
#----------- prior --------
priors = [
"Uniform",
# "Cauchy",
"Gaussian",
"GMM",
"EFF",
"King"
]
formatters={"Uniform_min":lambda x: "${0:4.0f}$".format(x)}


##################### LOOP OVER CLUSTERS ###################

for case in list_of_cases:
	#============ Directories and data =================
	dir_main   = "/home/javier/Repositories/Kalkayotl/"
	dir_out    = dir_main  + "Outputs/"
	dir_chains = dir_out   + case + "/"
	file_tex   = dir_out   + "Tables/Parameters_"+case+".tex"
	#=======================================================================================================================


	#--------------- First prior ---------------------------------------------------------------------------
	file_csv = dir_chains + priors[0]+"/Cluster_mode.csv"
	first    = pn.read_csv(file_csv,usecols=["Parameter","mode","lower","upper"])
	first.rename(index=str,inplace=True, 
			columns={"lower": priors[0]+"_min","mode": priors[0],"upper": priors[0]+"_max"})
	first.set_index("Parameter",inplace=True)

	#--------------- First evidence ---------------------------------------------------------------------------
	file_csv = dir_chains + priors[0]+"/Cluster_Z.csv"
	ev_0    = pn.read_csv(file_csv,usecols=["Parameter","median","lower","upper"])
	ev_0.rename(index=str,inplace=True, 
			columns={"lower": priors[0]+"_min","median": priors[0],"upper": priors[0]+"_max"})
	ev_0.set_index("Parameter",inplace=True)

	first = first.append(ev_0.loc["logZ"])


	#---------- Rest of the priors -----------------------------
	for prior in priors[1:]:
		#------ Read modes --------------------------------------------------------
		file_csv = dir_chains + prior+"/Cluster_mode.csv"
		rest     = pn.read_csv(file_csv,usecols=["Parameter","mode","lower","upper"])
		rest.rename(index=str,inplace=True, 
			columns={"lower": prior+"_min","mode": prior,"upper": prior+"_max"})
		rest.set_index("Parameter",inplace=True)

		#------ Read evidence --------------------------------------------------------
		file_csv = dir_chains + prior+"/Cluster_Z.csv"
		ev_rst     = pn.read_csv(file_csv,usecols=["Parameter","median","lower","upper"])
		ev_rst.rename(index=str,inplace=True, 
			columns={"lower": prior+"_min","median": prior,"upper": prior+"_max"})
		ev_rst.set_index("Parameter",inplace=True)

		rest = rest.append(ev_rst.loc["logZ"])

		#------------ Merge with data ---------------------------------------------------------
		first     = first.merge(rest,how="outer",left_index=True, right_index=True,suffixes=("_l","_r"))
	#------- Write tex file ---------------
	with open(file_tex, "w") as tex_file:
		header = sum([[first.index.name],[col for col in priors]],[])
		print("\\begin{tabular}{c"+len(priors)*"l"+"}", file=tex_file)
		print("\\hline", file=tex_file)
		print("\\hline", file=tex_file)
		print("  &  ".join(header) + "  \\\\", file=tex_file)
		print("\\hline", file=tex_file)
		for index,row in first.iterrows():
			str_cols = [index.replace("flavour_1d_","").replace("__","\\_")]
			for prior in priors:
				ctr = row[prior]
				ul  = ctr - row[prior+"_min"]
				uu  = row[prior+"_max"] - ctr
				if np.isnan(ctr):
					str_col = "                   "
				else:
					str_col = "${0:0.1f}_{{-{1:0.1f}}}^{{+{2:0.1f}}}$".format(ctr,ul,uu)
				str_cols.append(str_col)
			print("  &  ".join(str_cols) + "  \\\\", file=tex_file)
		print("\\hline", file=tex_file)
		print("\\end{tabular}", file=tex_file)
	print("Check file: ",file_tex)