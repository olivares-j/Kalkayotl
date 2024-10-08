import sys
import os
import numpy as np

#---------- Directory of user -------
dir_local = "/home/jolivares/"
dir_host = "/home/jromero/"

if dir_local in os.getcwd():
	dir_usr = dir_local
	dir_out = dir_usr + "Dropbox/MisArticulos/Kalkayotl/"
	dir_rep = dir_usr + "Repos/"
	dir_apo = dir_usr + "OCs/Catalogues/APOGEE/"
	dir_iso = dir_usr + ".isochrones/"
elif dir_host in os.getcwd():
	dir_usr = dir_host
	dir_out = dir_usr + "Projects/Kalkayotl/"
	dir_rep = "/home/jromero/Repos/"
	dir_apo = dir_usr + "OCs/Catalogues/APOGEE/"
	dir_iso = dir_usr + ".isochrones/"
else:
	sys.exit("User directory not identified!")
#--------------------------------

#------------------ Directories -----------------
dir_main = dir_usr + "Projects/Kalkayotl/"
dir_cat  = dir_usr + "Projects/Kalkayotl/Catalogues/"
dir_res  = dir_usr + "Projects/Kalkayotl/Results/"
dir_syn  = dir_usr + "Projects/Kalkayotl/Synthetic/"
dir_mem  = dir_usr + "Projects/Kalkayotl/Mecayotl/"
dir_kal  = dir_rep + "Kalkayotl/"
dir_ama  = dir_rep + "Amasijo/"
dir_mcm  = dir_rep + "McMichael/"
dir_mec  = dir_rep + "Mecayotl/"
dir_fig  = dir_out + "Figures/"
dir_tab  = dir_out + "Tables/"
#------------------------------------------------

file_apogee = dir_apo + "allStar-dr17-synspec_rev1.fits"
file_gaia   = dir_cat + "MOVeS_200pc.fits"
file_moves  = dir_cat + "MOVeS_200pc_GA.fits"

#=============================================== BALYSA ==========================================================================================================================================
parallax_limits = {"min":5.0,"max":np.inf}
GROUPS = {
"BPIC":{
	"authors": ["LACEwING","BANYAN","Lee+2019","Crundall+2019","Miret-Roig+2020","Miret-Roig+2020","SPYGLASS","Moranta+2022","Couture+2023","Couture+2023","UNION","Halo","Core"],#"Mecayotl_1"],
	"cases":   ["GAS"     ,"GAS"   ,"GAS"     ,"GAS"          ,"GAS"            ,"original"       ,"GAS"     ,"GAS"         ,"GAS"         ,"original"    ,"GAS"  ,"GAS" ,"GAS" ],#"GAS"       ],
	"GMM":     [False     ,False   ,False     ,False          ,False            ,False            ,False     ,False         ,False         ,False         ,True   ,False ,False ],#False       ],
	"maha":    [10        ,10      ,10        ,10             ,10               ,10               ,10        ,10            ,10            ,10            ,10     ,10    ,10    ],#10          ],
	"Mecayotl":[False     ,False   ,False     ,False          ,False            ,True             ,False     ,False         ,False         ,True          ,False  ,False ,True  ],#False       ],
	"field_scale":[40.,40.,40.,2.,2.,2.],
	"sky_error_factor":1e7,
	"final_origin":["Core","Halo"],
	"GMM_a":np.array([3,7]),
	"age":{"loc":23.,"scl":8.,}, #Lee 2024
	"literature":[
				{"author":"Barrado y Navascués et al. (1999)","age":20. ,"lower":10    ,"upper":10.   ,"method":"Isochrone"},
				{"author":"Zuckerman et al. (2001)"          ,"age":12. ,"lower":4     ,"upper":8.    ,"method":"Isochrone"},
				{"author":"Ortega et al. (2002)"             ,"age":11.5,"lower":10    ,"upper":10.   ,"method":"Traceback"},
				{"author":"Song et al. (2003)"               ,"age":12. ,"lower":np.nan,"upper":np.nan,"method":"Traceback"},
				{"author":"Ortega et al. (2004)"             ,"age":10.8,"lower":0.3   ,"upper":0.3   ,"method":"Traceback"},
				{"author":"Torres et al. (2006)"             ,"age":18. ,"lower":np.nan,"upper":np.nan,"method":"Expansion"},
				{"author":"Makarov (2007)"                   ,"age":31. ,"lower":21    ,"upper":21.   ,"method":"Traceback"},
				{"author":"Mentuch et al. (2008)"            ,"age":21. ,"lower":9.    ,"upper":9.    ,"method":"LDB"},
				{"author":"Macdonald & Mullan (2010)"        ,"age":40. ,"lower":np.nan,"upper":np.nan,"method":"LDB"},
				{"author":"Binks & Jeffries (2014)"          ,"age":21. ,"lower":4     ,"upper":4.    ,"method":"LDB"},
				{"author":"Malo et al. (2014)"               ,"age":26. ,"lower":3     ,"upper":3.    ,"method":"LDB"},
				{"author":"Malo et al. (2014)"               ,"age":21.5,"lower":6.5   ,"upper":6.5   ,"method":"Isochrone"},
				{"author":"Mamajek & Bell (2014)"            ,"age":22. ,"lower":3     ,"upper":3.    ,"method":"Isochrone"},
				{"author":"Mamajek & Bell (2014)"            ,"age":21. ,"lower":5     ,"upper":10    ,"method":"Expansion"},
				{"author":"Bell et al. (2015)"               ,"age":24. ,"lower":3     ,"upper":3.    ,"method":"Isochrone"},
				{"author":"Messina et al. (2016)"            ,"age":25. ,"lower":3     ,"upper":3.    ,"method":"LDB"},
				{"author":"Miret-Roig et al. (2018)"         ,"age":13. ,"lower":0     ,"upper":7.    ,"method":"Traceback"},
				{"author":"Crundall et al. (2019)"           ,"age":18.3,"lower":1.2   ,"upper":1.3   ,"method":"Forward-modelling"},
				{"author":"Ujjwal et al. (2020)"             ,"age":19.4,"lower":13.8  ,"upper":35.1  ,"method":"Isochrone"},
				{"author":"Miret-Roig et al. (2020)"         ,"age":18.5,"lower":2.4   ,"upper":2.0   ,"method":"Traceback"},
				{"author":"Galindo-Guil et al. (2022)"       ,"age":24.3,"lower":0.3   ,"upper":0.3   ,"method":"LDB"},
				{"author":"Couture et al. (2023)"            ,"age":20.4,"lower":2.5   ,"upper":2.5   ,"method":"Traceback"},
				{"author":"Lee et al. (2024)"                ,"age":33  ,"lower":9     ,"upper":11    ,"method":"Isochrone"},
				{"author":"Lee et al. (2024)"                ,"age":23  ,"lower":8     ,"upper":8     ,"method":"LDB"},
				{"author":"This work"                        ,"age":25.8,"lower":5.    ,"upper":5.   ,"method":"Expansion",
				"file_posterior":"{0}/Results/BPIC/Core_GAS/Gaussian_age_GGL_GE_10/Chains.nc".format(dir_main)},
				],
	"figsize":(6,8)
	},
"TWA":{
	"authors": ["LACEwING","BANYAN","Lee+2019","Luhman2023","Luhman2023","Miret-Roig+2024","Miret-Roig+2024","UNION"],#,"Core","Halo", "Miret-Roig+2023","Miret-Roig+2023"
	"cases":   ["GAS"     ,"GAS"   ,"GAS"     ,"GAS"       ,"original"  ,"GAS"            ,"original"       ,"GAS"  ],#, "GAS","GAS" , "original"       ,"GAS"            
	"GMM":     [False     ,False   ,False     ,False       ,False       ,True             ,True             ,True   ],#, False,False , False            ,False            
	"maha":    [10        ,10      ,10        ,10          ,10          ,10               ,10               ,10     ],#,    10,10    , 10               ,10               
	"Mecayotl":[False     ,False   ,False     ,False       ,False       ,False            ,False            ,False  ],#, True ,False , False            ,False            
	"field_scale":[20.,20.,20.,5.,5.,5.],
	"sky_error_factor":1e6,
	"final_origin":["UNION"],
	"GMM_a":[3,7],
	"age":{"loc":10.,"scl":5.}, # 10 Luhman2023
	"literature":[
				{"author":"Makarov et al. (2005)"         ,"age":4.7  ,"lower":0.6     ,"upper":0.6     ,"method":"Expansion"},
				{"author":"de la Reza et al. (2006)"      ,"age":8.3  ,"lower":0.8     ,"upper":0.8     ,"method":"Traceback"},
				{"author":"Barrado y Navascues (2006)"    ,"age":8.   ,"lower":5       ,"upper":12      ,"method":"Isochrone"},
				{"author":"Barrado y Navascues (2006)"    ,"age":8.   ,"lower":5       ,"upper":7.      ,"method":"LDB"},
				{"author":"Torres et al. (2008)"          ,"age":8.   ,"lower":np.nan  ,"upper":np.nan  ,"method":"Isochrone"},
				{"author":"Weinberger et al. (2013)"      ,"age":8.7  ,"lower":3.3     ,"upper":3.3     ,"method":"Isochrone"},
				{"author":"Ducourant et al. (2014)"       ,"age":7.5  ,"lower":0.7     ,"upper":0.7     ,"method":"Traceback"},
				{"author":"Bell et al. (2015)"            ,"age":10.  ,"lower":3       ,"upper":3.      ,"method":"Isochrone"},
				{"author":"Donaldson et al. (2016)"       ,"age":3.8  ,"lower":1.1     ,"upper":1.1     ,"method":"Traceback"},
				{"author":"Donaldson et al. (2016)"       ,"age":7.9  ,"lower":1.0     ,"upper":1.0     ,"method":"Isochrone"},
				{"author":"Luhman (2023)"                 ,"age":11.4 ,"lower":1.2     ,"upper":1.3     ,"method":"Isochrone"},
				{"author":"Luhman (2023)"                 ,"age":9.6  ,"lower":0.9     ,"upper":0.8     ,"method":"Expansion"},
				{"author":"This work"                     ,"age":10.3 ,"lower":1.1     ,"upper":1.1     ,"method":"Expansion",
				"file_posterior":"{0}/Results/TWA/UNION_GAS/Gaussian_age_GGL_GE_10/Chains.nc".format(dir_main)}
				],
	"figsize":(6,6)
	},
}

#=======================================================================================================================================================


true_signal  = 0.1
true_pos_sds = np.array([3.,3.,3.])
true_vel_loc = np.array([10.,10.,10.])
true_vel_sds = np.array([1.,1.,1.])
true_kappa   = np.ones(3)*true_signal
true_omega   = np.array([[-1,1,-1],[1,-1,1]])*true_signal
true_weights = np.array([0.6,0.4])
true_nu      = 10.

list_of_distances = [50,100,200,400,800,1000,1500]
list_of_n_stars   = [100,200,400]
list_of_seeds     = [0,1,2,3,4,5]
#list_of_signals   = [10,50,100]

# mass_limits = {
# 	50:[0.1,2.8],
# 	100:[0.1,4.3],
# 	200:[0.1,4.3],
# 	400:[0.1,5.],
# 	800:[0.2,5.],
# 	1000:[0.5,5.]
# }

PRIOR = {
"GMM_joint":{"type":"GMM",
		"parameters":{"location":None,
					  "scale":None,
					  "weights":None
					  },
		"hyper_parameters":{
							"location":None,
							"scale":{"loc":np.hstack([true_pos_sds,true_vel_sds])}, 
							"weights":{"n_components":2,"a":true_weights*10},
							"eta":None
							},
		},
"Gaussian_joint":{"type":"Gaussian",
		"parameters":{"location":None,"scale":None},
		"hyper_parameters":{
							"location":None,
							"scale":{"loc":np.hstack([true_pos_sds,true_vel_sds])},  
							"eta":None
							},
		},
"StudentT_joint":{"type":"StudentT",
		"parameters":{"location":None,"scale":None,"nu":None},
		"hyper_parameters":{
							"location":None,
							"scale":{"loc":np.hstack([true_pos_sds,true_vel_sds])},  
							"eta":None,
							"nu":{"alpha":1.0,"beta":1/10.}
							},
		},
"StudentT_linear":{"type":"StudentT",
		"parameters":{"location":None,"scale":None,"nu":None,"kappa":None,"omega":None},
		"hyper_parameters":{
							"location":None,
							"scale":{"loc":np.hstack([true_pos_sds,true_vel_sds])},  
							"eta":None,
							"kappa":None,
							"omega":None,
							"nu":{"alpha":1.0,"beta":1/10.}
							},
		},
"Gaussian_linear":{"type":"Gaussian",
		"parameters":{"location":None,"scale":None,"kappa":None,"omega":None},
		"hyper_parameters":{
							"location":None,
							"scale":{"loc":np.hstack([true_pos_sds,true_vel_sds])}, 
							"eta":None,
							"kappa":None,
							"omega":None
							},
		}
}

CASE_ARGS = {
"GMM_joint_d50":{
			"position+velocity":{"family":"GMM",
							"location":np.array([
								np.concatenate([np.repeat(50./np.sqrt(3),3),true_vel_loc]),
								np.concatenate([np.repeat(50./np.sqrt(3),3)+np.array([0.,0.,50.]),true_vel_loc])]),
							"covariance":np.array([
								np.diag(np.concatenate([true_pos_sds**2,true_vel_sds**2])),
								np.diag(np.concatenate([true_pos_sds**2,true_vel_sds**2]))]),
							"weights":true_weights
							},
			"true_parameters":{
					"6D::loc[A, X]":50./np.sqrt(3),
					"6D::loc[A, Y]":50./np.sqrt(3),
					"6D::loc[A, Z]":50./np.sqrt(3),
					"6D::loc[A, U]":true_vel_loc[0],
					"6D::loc[A, V]":true_vel_loc[1],
					"6D::loc[A, W]":true_vel_loc[2],
					"6D::std[A, X]":true_pos_sds[0],
					"6D::std[A, Y]":true_pos_sds[1],
					"6D::std[A, Z]":true_pos_sds[2],
					"6D::std[A, U]":true_vel_sds[0],
					"6D::std[A, V]":true_vel_sds[1],
					"6D::std[A, W]":true_vel_sds[2],
					"6D::loc[B, X]":50./np.sqrt(3),
					"6D::loc[B, Y]":50./np.sqrt(3),
					"6D::loc[B, Z]":50./np.sqrt(3)+50.,
					"6D::loc[B, U]":true_vel_loc[0],
					"6D::loc[B, V]":true_vel_loc[1],
					"6D::loc[B, W]":true_vel_loc[2],
					"6D::std[B, X]":true_pos_sds[0],
					"6D::std[B, Y]":true_pos_sds[1],
					"6D::std[B, Z]":true_pos_sds[2],
					"6D::std[B, U]":true_vel_sds[0],
					"6D::std[B, V]":true_vel_sds[1],
					"6D::std[B, W]":true_vel_sds[2],
					"6D::weights[A]":true_weights[0],
					"6D::weights[B]":true_weights[1],
							},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,2.8], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"GMM_joint_d100":{
			"position+velocity":{"family":"GMM",
							"location":np.array([
								np.concatenate([np.repeat(100./np.sqrt(3),3),true_vel_loc]),
								np.concatenate([np.repeat(100./np.sqrt(3),3)+np.array([0.,0.,50.]),true_vel_loc])]),
							"covariance":np.array([
								np.diag(np.concatenate([true_pos_sds**2,true_vel_sds**2])),
								np.diag(np.concatenate([true_pos_sds**2,true_vel_sds**2]))]),
							"weights":true_weights
							},
			"true_parameters":{
					"6D::loc[A, X]":100./np.sqrt(3),
					"6D::loc[A, Y]":100./np.sqrt(3),
					"6D::loc[A, Z]":100./np.sqrt(3),
					"6D::loc[A, U]":true_vel_loc[0],
					"6D::loc[A, V]":true_vel_loc[1],
					"6D::loc[A, W]":true_vel_loc[2],
					"6D::std[A, X]":true_pos_sds[0],
					"6D::std[A, Y]":true_pos_sds[1],
					"6D::std[A, Z]":true_pos_sds[2],
					"6D::std[A, U]":true_vel_sds[0],
					"6D::std[A, V]":true_vel_sds[1],
					"6D::std[A, W]":true_vel_sds[2],
					"6D::loc[B, X]":100./np.sqrt(3),
					"6D::loc[B, Y]":100./np.sqrt(3),
					"6D::loc[B, Z]":100./np.sqrt(3)+50.,
					"6D::loc[B, U]":true_vel_loc[0],
					"6D::loc[B, V]":true_vel_loc[1],
					"6D::loc[B, W]":true_vel_loc[2],
					"6D::std[B, X]":true_pos_sds[0],
					"6D::std[B, Y]":true_pos_sds[1],
					"6D::std[B, Z]":true_pos_sds[2],
					"6D::std[B, U]":true_vel_sds[0],
					"6D::std[B, V]":true_vel_sds[1],
					"6D::std[B, W]":true_vel_sds[2],
					"6D::weights[A]":true_weights[0],
					"6D::weights[B]":true_weights[1],
							},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,4.3], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"GMM_joint_d200":{
			"position+velocity":{"family":"GMM",
							"location":np.array([
								np.concatenate([np.repeat(200./np.sqrt(3),3),true_vel_loc]),
								np.concatenate([np.repeat(200./np.sqrt(3),3)+np.array([0.,0.,50.]),true_vel_loc])]),
							"covariance":np.array([
								np.diag(np.concatenate([true_pos_sds**2,true_vel_sds**2])),
								np.diag(np.concatenate([true_pos_sds**2,true_vel_sds**2]))]),
							"weights":true_weights
							},
			"true_parameters":{
					"6D::loc[A, X]":200./np.sqrt(3),
					"6D::loc[A, Y]":200./np.sqrt(3),
					"6D::loc[A, Z]":200./np.sqrt(3),
					"6D::loc[A, U]":true_vel_loc[0],
					"6D::loc[A, V]":true_vel_loc[1],
					"6D::loc[A, W]":true_vel_loc[2],
					"6D::std[A, X]":true_pos_sds[0],
					"6D::std[A, Y]":true_pos_sds[1],
					"6D::std[A, Z]":true_pos_sds[2],
					"6D::std[A, U]":true_vel_sds[0],
					"6D::std[A, V]":true_vel_sds[1],
					"6D::std[A, W]":true_vel_sds[2],
					"6D::loc[B, X]":200./np.sqrt(3),
					"6D::loc[B, Y]":200./np.sqrt(3),
					"6D::loc[B, Z]":200./np.sqrt(3)+50.,
					"6D::loc[B, U]":true_vel_loc[0],
					"6D::loc[B, V]":true_vel_loc[1],
					"6D::loc[B, W]":true_vel_loc[2],
					"6D::std[B, X]":true_pos_sds[0],
					"6D::std[B, Y]":true_pos_sds[1],
					"6D::std[B, Z]":true_pos_sds[2],
					"6D::std[B, U]":true_vel_sds[0],
					"6D::std[B, V]":true_vel_sds[1],
					"6D::std[B, W]":true_vel_sds[2],
					"6D::weights[A]":true_weights[0],
					"6D::weights[B]":true_weights[1],
							},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,4.3], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"GMM_joint_d400":{
			"position+velocity":{"family":"GMM",
							"location":np.array([
								np.concatenate([np.repeat(400./np.sqrt(3),3),true_vel_loc]),
								np.concatenate([np.repeat(400./np.sqrt(3),3)+np.array([0.,0.,50.]),true_vel_loc])]),
							"covariance":np.array([
								np.diag(np.concatenate([true_pos_sds**2,true_vel_sds**2])),
								np.diag(np.concatenate([true_pos_sds**2,true_vel_sds**2]))]),
							"weights":true_weights
							},
			"true_parameters":{
					"6D::loc[A, X]":400./np.sqrt(3),
					"6D::loc[A, Y]":400./np.sqrt(3),
					"6D::loc[A, Z]":400./np.sqrt(3),
					"6D::loc[A, U]":true_vel_loc[0],
					"6D::loc[A, V]":true_vel_loc[1],
					"6D::loc[A, W]":true_vel_loc[2],
					"6D::std[A, X]":true_pos_sds[0],
					"6D::std[A, Y]":true_pos_sds[1],
					"6D::std[A, Z]":true_pos_sds[2],
					"6D::std[A, U]":true_vel_sds[0],
					"6D::std[A, V]":true_vel_sds[1],
					"6D::std[A, W]":true_vel_sds[2],
					"6D::loc[B, X]":400./np.sqrt(3),
					"6D::loc[B, Y]":400./np.sqrt(3),
					"6D::loc[B, Z]":400./np.sqrt(3)+50.,
					"6D::loc[B, U]":true_vel_loc[0],
					"6D::loc[B, V]":true_vel_loc[1],
					"6D::loc[B, W]":true_vel_loc[2],
					"6D::std[B, X]":true_pos_sds[0],
					"6D::std[B, Y]":true_pos_sds[1],
					"6D::std[B, Z]":true_pos_sds[2],
					"6D::std[B, U]":true_vel_sds[0],
					"6D::std[B, V]":true_vel_sds[1],
					"6D::std[B, W]":true_vel_sds[2],
					"6D::weights[A]":true_weights[0],
					"6D::weights[B]":true_weights[1],
							},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,5.], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"GMM_joint_d800":{
			"position+velocity":{"family":"GMM",
							"location":np.array([
								np.concatenate([np.repeat(800./np.sqrt(3),3),true_vel_loc]),
								np.concatenate([np.repeat(800./np.sqrt(3),3)+np.array([0.,0.,50.]),true_vel_loc])]),
							"covariance":np.array([
								np.diag(np.concatenate([true_pos_sds**2,true_vel_sds**2])),
								np.diag(np.concatenate([true_pos_sds**2,true_vel_sds**2]))]),
							"weights":true_weights
							},
			"true_parameters":{
					"6D::loc[A, X]":800./np.sqrt(3),
					"6D::loc[A, Y]":800./np.sqrt(3),
					"6D::loc[A, Z]":800./np.sqrt(3),
					"6D::loc[A, U]":true_vel_loc[0],
					"6D::loc[A, V]":true_vel_loc[1],
					"6D::loc[A, W]":true_vel_loc[2],
					"6D::std[A, X]":true_pos_sds[0],
					"6D::std[A, Y]":true_pos_sds[1],
					"6D::std[A, Z]":true_pos_sds[2],
					"6D::std[A, U]":true_vel_sds[0],
					"6D::std[A, V]":true_vel_sds[1],
					"6D::std[A, W]":true_vel_sds[2],
					"6D::loc[B, X]":800./np.sqrt(3),
					"6D::loc[B, Y]":800./np.sqrt(3),
					"6D::loc[B, Z]":800./np.sqrt(3)+50.,
					"6D::loc[B, U]":true_vel_loc[0],
					"6D::loc[B, V]":true_vel_loc[1],
					"6D::loc[B, W]":true_vel_loc[2],
					"6D::std[B, X]":true_pos_sds[0],
					"6D::std[B, Y]":true_pos_sds[1],
					"6D::std[B, Z]":true_pos_sds[2],
					"6D::std[B, U]":true_vel_sds[0],
					"6D::std[B, V]":true_vel_sds[1],
					"6D::std[B, W]":true_vel_sds[2],
					"6D::weights[A]":true_weights[0],
					"6D::weights[B]":true_weights[1],
							},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.2,5.], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"GMM_joint_d1000":{
			"position+velocity":{"family":"GMM",
							"location":np.array([
								np.concatenate([np.repeat(1000./np.sqrt(3),3),true_vel_loc]),
								np.concatenate([np.repeat(1000./np.sqrt(3),3)+np.array([0.,0.,50.]),true_vel_loc])]),
							"covariance":np.array([
								np.diag(np.concatenate([true_pos_sds**2,true_vel_sds**2])),
								np.diag(np.concatenate([true_pos_sds**2,true_vel_sds**2]))]),
							"weights":true_weights
							},
			"true_parameters":{
					"6D::loc[A, X]":1000./np.sqrt(3),
					"6D::loc[A, Y]":1000./np.sqrt(3),
					"6D::loc[A, Z]":1000./np.sqrt(3),
					"6D::loc[A, U]":true_vel_loc[0],
					"6D::loc[A, V]":true_vel_loc[1],
					"6D::loc[A, W]":true_vel_loc[2],
					"6D::std[A, X]":true_pos_sds[0],
					"6D::std[A, Y]":true_pos_sds[1],
					"6D::std[A, Z]":true_pos_sds[2],
					"6D::std[A, U]":true_vel_sds[0],
					"6D::std[A, V]":true_vel_sds[1],
					"6D::std[A, W]":true_vel_sds[2],
					"6D::loc[B, X]":1000./np.sqrt(3),
					"6D::loc[B, Y]":1000./np.sqrt(3),
					"6D::loc[B, Z]":1000./np.sqrt(3)+50.,
					"6D::loc[B, U]":true_vel_loc[0],
					"6D::loc[B, V]":true_vel_loc[1],
					"6D::loc[B, W]":true_vel_loc[2],
					"6D::std[B, X]":true_pos_sds[0],
					"6D::std[B, Y]":true_pos_sds[1],
					"6D::std[B, Z]":true_pos_sds[2],
					"6D::std[B, U]":true_vel_sds[0],
					"6D::std[B, V]":true_vel_sds[1],
					"6D::std[B, W]":true_vel_sds[2],
					"6D::weights[A]":true_weights[0],
					"6D::weights[B]":true_weights[1],
							},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.5,5.], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_joint_d50":{
			"position+velocity":{"family":"Gaussian",
							"location":np.hstack([np.repeat(50./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":50./np.sqrt(3),
					"6D::loc[Y]":50./np.sqrt(3),
					"6D::loc[Z]":50./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,2.8], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_joint_d100":{
			"position+velocity":{"family":"Gaussian",
							"location":np.hstack([np.repeat(100./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":100./np.sqrt(3),
					"6D::loc[Y]":100./np.sqrt(3),
					"6D::loc[Z]":100./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,4.3], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_joint_d200":{
			"position+velocity":{"family":"Gaussian",
							"location":np.hstack([np.repeat(200./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":200./np.sqrt(3),
					"6D::loc[Y]":200./np.sqrt(3),
					"6D::loc[Z]":200./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,4.3], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_joint_d400":{
			"position+velocity":{"family":"Gaussian",
							"location":np.hstack([np.repeat(400./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":400./np.sqrt(3),
					"6D::loc[Y]":400./np.sqrt(3),
					"6D::loc[Z]":400./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,5.], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_joint_d800":{
			"position+velocity":{"family":"Gaussian",
							"location":np.hstack([np.repeat(800./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":800./np.sqrt(3),
					"6D::loc[Y]":800./np.sqrt(3),
					"6D::loc[Z]":800./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.2,5.], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_joint_d1000":{
			"position+velocity":{"family":"Gaussian",
							"location":np.hstack([np.repeat(1000./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":1000./np.sqrt(3),
					"6D::loc[Y]":1000./np.sqrt(3),
					"6D::loc[Z]":1000./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.5,5.0], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_joint_d1500":{
			"position+velocity":{"family":"Gaussian",
							"location":np.hstack([np.repeat(1500./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":1500./np.sqrt(3),
					"6D::loc[Y]":1500./np.sqrt(3),
					"6D::loc[Z]":1500./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.5,5.0], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_joint_d50":{
			"position+velocity":{"family":"Gaussian",
							"nu":true_nu,
							"location":np.hstack([np.repeat(50./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":50./np.sqrt(3),
					"6D::loc[Y]":50./np.sqrt(3),
					"6D::loc[Z]":50./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::nu":true_nu
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,2.8], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_joint_d100":{
			"position+velocity":{"family":"Gaussian",
							"nu":true_nu,
							"location":np.hstack([np.repeat(100./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":100./np.sqrt(3),
					"6D::loc[Y]":100./np.sqrt(3),
					"6D::loc[Z]":100./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::nu":true_nu
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,4.3], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_joint_d200":{
			"position+velocity":{"family":"Gaussian",
							"nu":true_nu,
							"location":np.hstack([np.repeat(200./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":200./np.sqrt(3),
					"6D::loc[Y]":200./np.sqrt(3),
					"6D::loc[Z]":200./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::nu":true_nu
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,4.3], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_joint_d400":{
			"position+velocity":{"family":"Gaussian",
							"nu":true_nu,
							"location":np.hstack([np.repeat(400./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":400./np.sqrt(3),
					"6D::loc[Y]":400./np.sqrt(3),
					"6D::loc[Z]":400./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::nu":true_nu
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,5.], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_joint_d800":{
			"position+velocity":{"family":"Gaussian",
							"nu":true_nu,
							"location":np.hstack([np.repeat(800./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":800./np.sqrt(3),
					"6D::loc[Y]":800./np.sqrt(3),
					"6D::loc[Z]":800./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::nu":true_nu
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.2,5.], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_joint_d1000":{
			"position+velocity":{"family":"Gaussian",
							"nu":true_nu,
							"location":np.hstack([np.repeat(1000./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":1000./np.sqrt(3),
					"6D::loc[Y]":1000./np.sqrt(3),
					"6D::loc[Z]":1000./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::nu":true_nu
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.5,5.0], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_joint_d1500":{
			"position+velocity":{"family":"Gaussian",
							"nu":true_nu,
							"location":np.hstack([np.repeat(1500./np.sqrt(3),3),true_vel_loc]),
							"covariance":np.hstack([true_pos_sds**2,true_vel_sds**2])},
			"true_parameters":{
					"6D::loc[X]":1500./np.sqrt(3),
					"6D::loc[Y]":1500./np.sqrt(3),
					"6D::loc[Z]":1500./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::nu":true_nu
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.5,5.0], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_linear_d50":{
			"position":{"family":"Gaussian",
							"location":np.repeat(50./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"Gaussian",
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":50./np.sqrt(3),
					"6D::loc[Y]":50./np.sqrt(3),
					"6D::loc[Z]":50./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,2.7], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_linear_d100":{
			"position":{"family":"Gaussian",
							"location":np.repeat(100./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"Gaussian",
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":100./np.sqrt(3),
					"6D::loc[Y]":100./np.sqrt(3),
					"6D::loc[Z]":100./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,4.3], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_linear_d200":{
			"position":{"family":"Gaussian",
							"location":np.repeat(200./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"Gaussian",
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":200./np.sqrt(3),
					"6D::loc[Y]":200./np.sqrt(3),
					"6D::loc[Z]":200./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,4.3], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_linear_d400":{
			"position":{"family":"Gaussian",
							"location":np.repeat(400./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"Gaussian",
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":400./np.sqrt(3),
					"6D::loc[Y]":400./np.sqrt(3),
					"6D::loc[Z]":400./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,5.0], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_linear_d800":{
			"position":{"family":"Gaussian",
							"location":np.repeat(800./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"Gaussian",
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":800./np.sqrt(3),
					"6D::loc[Y]":800./np.sqrt(3),
					"6D::loc[Z]":800./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.2,5.0], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_linear_d1000":{
			"position":{"family":"Gaussian",
							"location":np.repeat(1000./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"Gaussian",
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":1000./np.sqrt(3),
					"6D::loc[Y]":1000./np.sqrt(3),
					"6D::loc[Z]":1000./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.5,5.0], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"Gaussian_linear_d1500":{
			"position":{"family":"Gaussian",
							"location":np.repeat(1500./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"Gaussian",
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":1500./np.sqrt(3),
					"6D::loc[Y]":1500./np.sqrt(3),
					"6D::loc[Z]":1500./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2]
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.5,5.0], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_linear_d50":{
			"position":{"family":"StudentT",
							"nu":true_nu,
							"location":np.repeat(50./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"StudentT",
							"nu":true_nu,
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":50./np.sqrt(3),
					"6D::loc[Y]":50./np.sqrt(3),
					"6D::loc[Z]":50./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2],
					"6D::nu[0]":true_nu,
					"6D::nu[1]":true_nu,
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,2.8], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_linear_d100":{
			"position":{"family":"StudentT",
							"nu":true_nu,
							"location":np.repeat(100./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"StudentT",
							"nu":true_nu,
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":100./np.sqrt(3),
					"6D::loc[Y]":100./np.sqrt(3),
					"6D::loc[Z]":100./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2],
					"6D::nu[0]":true_nu,
					"6D::nu[1]":true_nu,
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,4.3], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_linear_d200":{
			"position":{"family":"StudentT",
							"nu":true_nu,
							"location":np.repeat(200./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"StudentT",
							"nu":true_nu,
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":200./np.sqrt(3),
					"6D::loc[Y]":200./np.sqrt(3),
					"6D::loc[Z]":200./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2],
					"6D::nu[0]":true_nu,
					"6D::nu[1]":true_nu,
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,4.3], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_linear_d400":{
			"position":{"family":"StudentT",
							"nu":true_nu,
							"location":np.repeat(400./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"StudentT",
							"nu":true_nu,
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":400./np.sqrt(3),
					"6D::loc[Y]":400./np.sqrt(3),
					"6D::loc[Z]":400./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2],
					"6D::nu[0]":true_nu,
					"6D::nu[1]":true_nu,
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.1,5.0], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_linear_d800":{
			"position":{"family":"StudentT",
							"nu":true_nu,
							"location":np.repeat(800./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"StudentT",
							"nu":true_nu,
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":800./np.sqrt(3),
					"6D::loc[Y]":800./np.sqrt(3),
					"6D::loc[Z]":800./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2],
					"6D::nu[0]":true_nu,
					"6D::nu[1]":true_nu,
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.2,5.0], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_linear_d1000":{
			"position":{"family":"StudentT",
							"nu":true_nu,
							"location":np.repeat(1000./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"StudentT",
							"nu":true_nu,
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":1000./np.sqrt(3),
					"6D::loc[Y]":1000./np.sqrt(3),
					"6D::loc[Z]":1000./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2],
					"6D::nu[0]":true_nu,
					"6D::nu[1]":true_nu,
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.5,5.0], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
"StudentT_linear_d1500":{
			"position":{"family":"StudentT",
							"nu":true_nu,
							"location":np.repeat(1500./np.sqrt(3),3),
							"covariance":np.diag(true_pos_sds**2)},
			"velocity":{"family":"StudentT",
							"nu":true_nu,
							"location":true_vel_loc,
							"covariance":np.diag(true_vel_sds**2),
							"kappa":true_kappa,
							"omega":true_omega
							},
			"true_parameters":{
					"6D::loc[X]":1500./np.sqrt(3),
					"6D::loc[Y]":1500./np.sqrt(3),
					"6D::loc[Z]":1500./np.sqrt(3),
					"6D::loc[U]":true_vel_loc[0],
					"6D::loc[V]":true_vel_loc[1],
					"6D::loc[W]":true_vel_loc[2],
					"6D::std[X]":true_pos_sds[0],
					"6D::std[Y]":true_pos_sds[1],
					"6D::std[Z]":true_pos_sds[2],
					"6D::std[U]":true_vel_sds[0],
					"6D::std[V]":true_vel_sds[1],
					"6D::std[W]":true_vel_sds[2],
					"6D::kappa[X]":true_kappa[0],
					"6D::kappa[Y]":true_kappa[1],
					"6D::kappa[Z]":true_kappa[2],
					"6D::omega[0, 0]":true_omega[0,0],
					"6D::omega[0, 1]":true_omega[0,1],
					"6D::omega[0, 2]":true_omega[0,2],
					"6D::omega[1, 0]":true_omega[1,0],
					"6D::omega[1, 1]":true_omega[1,1],
					"6D::omega[1, 2]":true_omega[1,2],
					"6D::nu[0]":true_nu,
					"6D::nu[1]":true_nu,
					},
			"isochrones_args":{
					"log_age": 8.0,    
					"metallicity":0.012,
					"Av": 0.0,         
					"mass_limits":[0.5,5.0], 
					"bands":["G","BP","RP"],
					"mass_prior":"Uniform"
					}
			},
}

	

# "BPIC":{
# 	"kalkayotl_args":{
# 				"file":dir_res+"BPIC/Core_GAS/Gaussian_linear_10/Cluster_statistics.csv",
# 				# "file":dir_syn+"BPIC/Cluster_statistics.csv",
# 				"statistic":"mean",
# 				"replace":{
# 						"6D::age":ASSOCIATIONS["BPIC"]["age"]["loc"],
# 						"6D::kappa[X]":kpf(ASSOCIATIONS["BPIC"]["age"]["loc"]),
# 						"6D::kappa[Y]":kpf(ASSOCIATIONS["BPIC"]["age"]["loc"]),
# 						"6D::kappa[Z]":kpf(ASSOCIATIONS["BPIC"]["age"]["loc"]),
# 						"6D::omega[0, 0]":-0.04,
# 						"6D::omega[0, 1]": 0.04,
# 						"6D::omega[0, 2]":-0.04,
# 						"6D::omega[1, 0]":0.0,
# 						"6D::omega[1, 1]":0.0,
# 						"6D::omega[1, 2]":0.0,
# 						# "6D::std[X]":20.0,
# 						# "6D::std[Y]":20.0,
# 						# "6D::std[Z]":20.0,
# 						# "6D::std[U]":0.50,
# 						# "6D::std[V]":0.50,
# 						# "6D::std[W]":0.50,
# 						},
# 				},
# 	"isochrones_args":{
# 				"log_age": np.log10(ASSOCIATIONS["BPIC"]["age"]["loc"]*1.e6),    
# 				"metallicity":0.012,
# 				"Av": 0.0,         
# 				"mass_limits":[0.8,1.4], 
# 				"bands":["G","BP","RP"],
# 				"mass_prior":"Uniform"
# 				},
# 	"surveys":["Gaia_dr3"],#"Gaia_dr4"],
# 	"list_of_n_stars":[100,50,25],
# 	"g_mag_shift_for_uncertainty":{"astrometry":8.0,"spectroscopy":0.0},
# 	"fractions_rvs_obs":[1.0,0.5,0.25],
# 	"age_priors":["GGL_{0}_{1}".format(int(ASSOCIATIONS["BPIC"]["age"]["loc"]),int(ASSOCIATIONS["BPIC"]["age"]["scl"]))]
# 	# "age_priors":["GGL_23_8_0.1","GGL_23_8_1","GGL_23_8_2"]
# 	# "age_priors":["GGL_23_8_1","GGL_23_8_10","GGL_23_8_20"]
# 	},

# }


gmm_mapper = {
"6D::loc[A, X]":"6D::loc[X]",
"6D::loc[A, Y]":"6D::loc[Y]",
"6D::loc[A, Z]":"6D::loc[Z]",
"6D::loc[A, U]":"6D::loc[U]",
"6D::loc[A, V]":"6D::loc[V]",
"6D::loc[A, W]":"6D::loc[W]",
"6D::std[A, X]":"6D::std[X]",
"6D::std[A, Y]":"6D::std[Y]",
"6D::std[A, Z]":"6D::std[Z]",
"6D::std[A, U]":"6D::std[U]",
"6D::std[A, V]":"6D::std[V]",
"6D::std[A, W]":"6D::std[W]"
		}