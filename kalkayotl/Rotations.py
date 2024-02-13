import sys
import numpy as np
import pytensor
from pytensor import tensor as tt

#--------------------------- Rotation from cluster to Galactic -----------------------------
# Quaternions
def np_quaternions_rotation_matrix(a,b,c,d):
	#r = tt.zeros((3,3))
	#r = np.zeros((3,3))

	r_0 = [1 - 2*(c**2 + d**2), 2*(b*c - a*d), 2*(b*d + a*c)]
	r_1 = [2*(b*c + a*d), 1 - 2*(b**2 + d**2), 2*(c*d - a*b)]
	r_2 = [2*(b*d - a*c), 2*(c*d + a*b), 1 - 2*(b**2 + c**2)]

	# r = tt.set_subtensor(r[0],r_0)
	# r = tt.set_subtensor(r[1],r_1)
	# r = tt.set_subtensor(r[2],r_2)

	r = [r_0, r_1, r_2]

	return r

def np_random_uniform_rotation_cluster_to_galactic(xyz, perezsala_parameters):
	theta1 = 2*np.pi*perezsala_parameters[1]
	theta2 = 2*np.pi*perezsala_parameters[2]
	r1 = np.sqrt(1 - perezsala_parameters[0])
	r2 = np.sqrt(perezsala_parameters[0])
	q = np_quaternions_rotation_matrix(np.cos(theta2)*r2, np.sin(theta1)*r1, np.cos(theta1)*r1, np.sin(theta2)*r2)	
	return q

def np_YZ_angle_rotation(xyz, angle):
	rot_matrix =  np.array([[1, 0, 0],
							[0, np.cos(angle), -np.sin(angle)],
							[0, np.sin(angle), np.cos(angle)]])
	print(rot_matrix)
	print(xyz[0])
	rotated = np.dot(rot_matrix, xyz.T)
	print(rotated.T[0])
	return rotated.T

def np_XZ_angle_rotation(xyz, angle):
	rot_matrix =  np.array([[np.cos(angle), 0, np.sin(angle)],
							[0, 1, 0],
							[-np.sin(angle), 0, np.cos(angle)]])
	print(rot_matrix)
	print(xyz[0])
	rotated = np.dot(rot_matrix, xyz.T)
	print(rotated.T[0])
	return rotated.T

def np_XY_angle_rotation(xyz, angle):
	rot_matrix =  np.array([[np.cos(angle), -np.sin(angle), 0],
							[np.sin(angle), np.cos(angle), 0],
							[0, 0, 1]])
	print(rot_matrix)
	print(xyz[0])
	rotated = np.dot(rot_matrix, xyz.T)
	print(rotated.T[0])
	return rotated.T

def np_angles3_rotation(xyz, angles):
	rotated_XY = np_XY_angle_rotation(xyz, angles[0])
	rotated_XZ = np_XZ_angle_rotation(rotated_XY, angles[1])
	rotated = np_YZ_angle_rotation(rotated_XZ, angles[2])
	return rotated

def np_translation_cluster_to_galactic_by_matrix(loc_galactic, tam=4):
    eye = np.eye(tam)
    loc_galactic = np.append(loc_galactic, 1)
    eye[:,tam-1] = loc_galactic
    return eye

def np_translation_cluster_to_galactic(perezsala_parameters, loc_galactic):
	return perezsala_parameters + loc_galactic

def np_cluster_to_galactic_by_matrix(xyz, perezsala_parameters, loc_galactic):
    q = np_random_uniform_rotation_cluster_to_galactic(xyz, perezsala_parameters)
    t = np_translation_cluster_to_galactic(loc_galactic)
    rotated = np.dot(q, xyz.T).T
    return np.dot(t, np.append(rotated, 1))[:-1]

def np_cluster_to_galactic(xyz, perezsala_parameters, loc_galactic):
    q = np_random_uniform_rotation_cluster_to_galactic(xyz, perezsala_parameters)
    rotated = np.dot(q, xyz.T).T
    return np_translation_cluster_to_galactic(rotated, loc_galactic)

def np_cluster_to_galactic_YZ(xyz, rot_angle, loc_galactic):
	rotated = np_YZ_angle_rotation(xyz, rot_angle)
	return np_translation_cluster_to_galactic(rotated, loc_galactic)

def np_cluster_to_galactic_XZ(xyz, rot_angle, loc_galactic):
	rotated = np_XZ_angle_rotation(xyz, rot_angle)
	return np_translation_cluster_to_galactic(rotated, loc_galactic)

def np_cluster_to_galactic_XY(xyz, rot_angle, loc_galactic):
	rotated = np_XY_angle_rotation(xyz, rot_angle)
	return np_translation_cluster_to_galactic(rotated, loc_galactic)

def np_cluster_to_galactic_3ang(xyz, rot_angles, loc_galactic):
	rotated = np_angles3_rotation(xyz, rot_angles)
	return np_translation_cluster_to_galactic(rotated, loc_galactic)

def quaternions_rotation_matrix(a,b,c,d):
	r = tt.zeros((3,3))

	r_0 = [1 - 2*(c**2 + d**2), 2*(b*c - a*d), 2*(b*d + a*c)]
	r_1 = [2*(b*c + a*d), 1 - 2*(b**2 + d**2), 2*(c*d - a*b)]
	r_2 = [2*(b*d - a*c), 2*(c*d + a*b), 1 - 2*(b**2 + c**2)]

	r = tt.set_subtensor(r[0],r_0)
	r = tt.set_subtensor(r[1],r_1)
	r = tt.set_subtensor(r[2],r_2)

	#r = tt.shared(np,array([r_0, r_1, r_2]))
	return r

def random_uniform_rotation_cluster_to_galactic(xyz, perezsala_parameters):
	theta1 = 2*np.pi*perezsala_parameters[1]
	theta2 = 2*np.pi*perezsala_parameters[2]
	r1 = tt.sqrt(1 - perezsala_parameters[0])
	r2 = tt.sqrt(perezsala_parameters[0])
	q = quaternions_rotation_matrix(tt.cos(theta2)*r2, tt.sin(theta1)*r1, tt.cos(theta1)*r1, tt.sin(theta2)*r2)
	
	return tt.dot(q, xyz.T).T

def YZ_angle_rotation(xyz, angle):
	r = tt.zeros(shape=(3,3))
	r_0 = tt.zeros((3,))
	r_1 = tt.zeros((3,))
	r_2 = tt.zeros((3,))
	r_0 = tt.set_subtensor(r_0[0], 1)
	r_0 = tt.set_subtensor(r_0[1], 0)
	r_0 = tt.set_subtensor(r_0[2], 0)
	r_1 = tt.set_subtensor(r_1[0], 0)
	r_1 = tt.set_subtensor(r_1[1], tt.cos(angle))
	r_1 = tt.set_subtensor(r_1[2], -tt.sin(angle))
	r_2 = tt.set_subtensor(r_2[0], 0)
	r_2 = tt.set_subtensor(r_2[1], tt.sin(angle))
	r_2 = tt.set_subtensor(r_2[2], tt.cos(angle))
	
	r = tt.set_subtensor(r[0], r_0)
	r = tt.set_subtensor(r[1],r_1)
	r = tt.set_subtensor(r[2],r_2)

	return tt.dot(r, xyz.T).T

def XZ_angle_rotation(xyz, angle):
	r = tt.zeros(shape=(3,3))
	r_0 = tt.zeros((3,))
	r_1 = tt.zeros((3,))
	r_2 = tt.zeros((3,))
	r_0 = tt.set_subtensor(r_0[0], tt.cos(angle))
	r_0 = tt.set_subtensor(r_0[1], 0)
	r_0 = tt.set_subtensor(r_0[2], tt.sin(angle))
	r_1 = tt.set_subtensor(r_1[0], 0)
	r_1 = tt.set_subtensor(r_1[1], 1)
	r_1 = tt.set_subtensor(r_1[2], 0)
	r_2 = tt.set_subtensor(r_2[0], -tt.sin(angle))
	r_2 = tt.set_subtensor(r_2[1], 0)
	r_2 = tt.set_subtensor(r_2[2], tt.cos(angle))
	
	r = tt.set_subtensor(r[0], r_0)
	r = tt.set_subtensor(r[1],r_1)
	r = tt.set_subtensor(r[2],r_2)

	return tt.dot(r, xyz.T).T

def XY_angle_rotation(xyz, angle):
	r = tt.zeros(shape=(3,3))
	r_0 = tt.zeros((3,))
	r_1 = tt.zeros((3,))
	r_2 = tt.zeros((3,))
	r_0 = tt.set_subtensor(r_0[0], tt.cos(angle))
	r_0 = tt.set_subtensor(r_0[1], -tt.sin(angle))
	r_0 = tt.set_subtensor(r_0[2], 0)
	r_1 = tt.set_subtensor(r_1[0], tt.sin(angle))
	r_1 = tt.set_subtensor(r_1[1], tt.cos(angle))
	r_1 = tt.set_subtensor(r_1[2], 0)
	r_2 = tt.set_subtensor(r_2[0], 0)
	r_2 = tt.set_subtensor(r_2[1], 0)
	r_2 = tt.set_subtensor(r_2[2], 1)
	
	r = tt.set_subtensor(r[0], r_0)
	r = tt.set_subtensor(r[1],r_1)
	r = tt.set_subtensor(r[2],r_2)

	return tt.dot(r, xyz.T).T

def angles3_rotation(xyz, angles):
	rotated_XY = XY_angle_rotation(xyz, angles[0])
	rotated_XZ = XZ_angle_rotation(rotated_XY, angles[1])
	rotated = YZ_angle_rotation(rotated_XZ, angles[2])
	return rotated

def translation_cluster_to_galactic(xyz, loc_galactic):
	return xyz + loc_galactic

def cluster_to_galactic(xyz, perezsala_parameters, loc_galactic):
    rotated = random_uniform_rotation_cluster_to_galactic(xyz, perezsala_parameters)
    return translation_cluster_to_galactic(rotated, loc_galactic)

def cluster_to_galactic_YZ(xyz, rot_angle, loc_galactic):
	rotated = YZ_angle_rotation(xyz, rot_angle)
	return translation_cluster_to_galactic(rotated, loc_galactic)

def cluster_to_galactic_XZ(xyz, rot_angle, loc_galactic):
	rotated = XZ_angle_rotation(xyz, rot_angle)
	return translation_cluster_to_galactic(rotated, loc_galactic)

def cluster_to_galactic_XY(xyz, rot_angle, loc_galactic):
	rotated = XY_angle_rotation(xyz, rot_angle)
	return translation_cluster_to_galactic(rotated, loc_galactic)

def cluster_to_galactic_3ang(xyz, rot_angles, loc_galactic):
	rotated = angles3_rotation(xyz, rot_angles)
	return translation_cluster_to_galactic(rotated, loc_galactic)

#-------------------------- PerezSala Parameters to Euler Angles ---------------------------

def np_perezsala_to_eulerangles(perezsala_parameters):
	theta1 = 2*np.pi*perezsala_parameters[1]
	theta2 = 2*np.pi*perezsala_parameters[2]
	r1 = np.sqrt(1 - perezsala_parameters[0])
	r2 = np.sqrt(perezsala_parameters[0])
	qw = np.cos(theta2)*r2
	qx = np.sin(theta1)*r1
	qy = np.cos(theta1)*r1
	qz = np.sin(theta2)*r2

	eulerangles_x = np.arctan2(2*(qw*qx + qy*qz),1-2*(qx**2 + qy**2))
	#eulerangles_y = -np.pi/2 + 2*np.arctan2(np.sqrt(1 + 2*(qw*qy - qx*qz)), np.sqrt(1 - 2*(qw*qy - qx*qz)))
	eulerangles_y = np.arcsin(2*(qw*qy - qx*qz))
	eulerangles_z = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy**2 + qz**2))

	return np.array([eulerangles_x, eulerangles_y, eulerangles_z])
	
def perezsala_to_eulerangles(perezsala_parameters):
	eulerangles = tt.zeros((3))

	theta1 = 2*np.pi*perezsala_parameters[1]
	theta2 = 2*np.pi*perezsala_parameters[2]
	r1 = tt.sqrt(1 - perezsala_parameters[0])
	r2 = tt.sqrt(perezsala_parameters[0])
	qw = tt.cos(theta2)*r2
	qx = tt.sin(theta1)*r1
	qy = tt.cos(theta1)*r1
	qz = tt.sin(theta2)*r2

	eulerangles_0 = tt.arctan2(2*(qw*qx + qy*qz),1-2*(qx**2 + qy**2))
	eulerangles_1 = tt.arcsin(2*(qw*qy - qx*qz))
	eulerangles_2 = tt.arctan2(2*(qw*qz + qx*qy), 1-2*(qy**2 + qz**2))

	eulerangles = tt.set_subtensor(eulerangles[0],eulerangles_0)
	eulerangles = tt.set_subtensor(eulerangles[1],eulerangles_1)
	eulerangles = tt.set_subtensor(eulerangles[2],eulerangles_2)

	return eulerangles

def np_eulerangles_to_perezsala(eulerangles):
	(yaw, pitch, roll) = (eulerangles[0], eulerangles[1], eulerangles[2])
	qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
	qz = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
	qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
	qx = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
	
	x2 = (np.arctan2(qx, qy) / (2*np.pi)) % 1
	x3 = (np.arctan2(qz, qw) / (2*np.pi)) % 1
	x1 = (qw/np.cos(2*np.pi*x3))**2 % 1
	
	return np.array([x1, x2, x3])

def compare_conversions():
	import math
	perezsala_parameters = np.random.uniform(size=(3))
	print(f'perezsala: {perezsala_parameters}')

	theta1 = 2*np.pi*perezsala_parameters[1]
	theta2 = 2*np.pi*perezsala_parameters[2]
	r1 = np.sqrt(1 - perezsala_parameters[0])
	r2 = np.sqrt(perezsala_parameters[0])
	qw = np.cos(theta2)*r2
	qx = np.sin(theta1)*r1
	qy = np.cos(theta1)*r1
	qz = np.sin(theta2)*r2

	print(f'qw: {qw} \n qx: {qx} \n qy: {qy} \n qz: {qz} \n')

	eulerangles = np_perezsala_to_eulerangles(perezsala_parameters)
	print(f'Euler angles [rad]: {eulerangles}')
	deg_angles = [math.degrees(eulerangles[0]), math.degrees(eulerangles[1]), math.degrees(eulerangles[2])]
	print(f'Euler angles [deg]: {deg_angles}')

	(yaw, pitch, roll) = (eulerangles[0], eulerangles[1], eulerangles[2])
	qw_ = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
	qx_ = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
	qy_ = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
	qz_ = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)

	print(f'qw: {qw_} \n qx: {qx_} \n qy: {qy_} \n qz: {qz_} \n')

	ps = np_eulerangles_to_perezsala(eulerangles)
	print(f'perezsala from Euler angles [rad]: {ps}')
	ang_ps = np_perezsala_to_eulerangles(ps)
	print(f'Euler angles from Euler angles [rad]: {ang_ps}')
	deg_ang_ps = [math.degrees(ang_ps[0]), math.degrees(ang_ps[1]), math.degrees(ang_ps[2])]
	print(f'Euler angles from Euler angles [deg]: {deg_ang_ps}')

def found_angley():
	import Transformations as tr
	import matplotlib.pyplot as plt
	import pandas as pd
	import math
	from sklearn.linear_model import LinearRegression
	members = pd.read_csv('article/v2.0/ComaBer/Core/members+rvs_tails.csv')
	res = tr.np_radecplx_to_galactic_xyz(np.array([[members.get('ra'), members.get('dec'), members.get('parallax')]]))[0]
	reg = LinearRegression()
	reg.fit(np.array([res[:,0]]).T, res[:,1])
	pred = reg.predict([[0],[20]])
	rot_ang = math.radians(90) - np.arctan2(np.abs(pred[1]-pred[0]), 20)
	#rot_ang = np.arctan(np.abs(pred[1]-pred[0]))
	print(reg.get_params())
	print(f'For points 0 and 1, Y value is: {pred}')
	print(f'The needed rotation angle is: {rot_ang} [rad] or {math.degrees(rot_ang)} [deg]')


def search_besty_rotation():
	import Transformations as tr
	import matplotlib.pyplot as plt
	import pandas as pd
	import math
	members = pd.read_csv('article/v2.0/ComaBer/Core/members+rvs_tails.csv')
	res = tr.np_radecplx_to_galactic_xyz(np.array([[members.get('ra'), members.get('dec'), members.get('parallax')]]))[0]


	#perezsala_parameters = np_eulerangles_to_perezsala(np.array([0, 0, math.radians(-40.20)]))
	perezsala_parameters = np_eulerangles_to_perezsala(np.array([0.20955699, 0.61366909, 0.95815134]))
	#perezsala_parameters = np_eulerangles_to_perezsala(np.array([0, 0, math.radians(-20.82)]))#np.random.uniform(size=(3))
	#perezsala_parameters = np_eulerangles_to_perezsala(np.array([0, 0, -0.726832924611887/2]))
	#perezsala_parameters = np_eulerangles_to_perezsala(np.array([-0.726832924611887, 0, 0]))
	#perezsala_parameters = np_eulerangles_to_perezsala(np.array([0, -0.726832924611887, 0]))
	print(f'perezsala: {perezsala_parameters}')
	angles = np_perezsala_to_eulerangles(perezsala_parameters)
	print(f'Euler angles [rad]: {angles}')
	deg_angles = [math.degrees(angles[0]), math.degrees(angles[1]), math.degrees(angles[2])]
	print(f'Euler angles [deg]: {deg_angles}')
	ps = np_eulerangles_to_perezsala(angles)
	print(f'perezsala from Euler angles [rad]: {ps}')
	ang_ps = np_perezsala_to_eulerangles(ps)
	print(f'Euler angles from Euler angles [rad]: {ang_ps}')
	deg_ang_ps = [math.degrees(ang_ps[0]), math.degrees(ang_ps[1]), math.degrees(ang_ps[2])]
	print(f'Euler angles from Euler angles [deg]: {deg_ang_ps}')

	res_rotated =  np_cluster_to_galactic(res, perezsala_parameters, loc_galactic, np.zeros(3))
	print(np.shape(res_rotated))

	ax1 = plt.subplot(2, 4, 1)
	plt.scatter(res[:,0], res[:,1], s=5)
	ax1.set_xlabel('X')
	ax1.set_ylabel('Y')
	ax1.set_xlim([-100,100])
	ax1.set_ylim([-90,90])

	ax2 = plt.subplot(2, 4, 2)
	plt.scatter(res[:,2], res[:,1], s=5)
	ax2.set_xlabel('Z')
	ax2.set_ylabel('Y')

	ax3 = plt.subplot(2, 4, 5)
	plt.scatter(res[:,0], res[:,2], s=5)
	ax3.set_xlabel('X')
	ax3.set_ylabel('Z')

	ax4 = plt.subplot(2, 4, 3)
	plt.scatter(res_rotated[:,0], res_rotated[:,1], s=5)
	ax4.set_xlabel('X')
	ax4.set_ylabel('Y')
	ax4.set_xlim([-100,100])
	ax4.set_ylim([-90,90])

	ax5 = plt.subplot(2, 4, 4)
	plt.scatter(res_rotated[:,2], res_rotated[:,1], s=5)
	ax5.set_xlabel('Z')
	ax5.set_ylabel('Y')

	ax6 = plt.subplot(2, 4, 7)
	plt.scatter(res_rotated[:,0], res_rotated[:,2], s=5)
	ax6.set_xlabel('X')
	ax6.set_ylabel('Z')
	plt.show()

def apply_compare_rotation(wich_plane:str='XY', centered:bool=True, is_np:bool=False):
	import Transformations as tr
	import matplotlib.pyplot as plt
	import pandas as pd
	import math
	members = pd.read_csv('article/v2.0/ComaBer/Core/members+rvs.csv')#pd.read_csv('article/v2.0/ComaBer/Core/members+rvs_tails.csv')
	res = tr.np_radecplx_to_galactic_xyz(np.array([[members.get('ra'), members.get('dec'), members.get('parallax')]]))[0]

	center_pos = np.mean(res, axis=0)
	if centered:
		res = res - center_pos

	rot_angle = math.radians(40)
	if wich_plane=='all':
		rot_angle = [math.radians(40), math.radians(10), math.radians(25)]

	if wich_plane=='YZ':
		if is_np:
			res_rotated = np_YZ_angle_rotation(res, rot_angle)
		else:
			f = pytensor.function([], YZ_angle_rotation(res, rot_angle))
			res_rotated = f()
	elif wich_plane=='XZ':
		if is_np:
			res_rotated = np_XZ_angle_rotation(res, rot_angle)
		else:
			f = pytensor.function([], XZ_angle_rotation(res, rot_angle))
			res_rotated = f()
	elif wich_plane=='XY':
		if is_np:
			res_rotated = np_XY_angle_rotation(res, rot_angle)
		else:
			f = pytensor.function([], XY_angle_rotation(res, rot_angle))
			res_rotated = f()
	elif wich_plane=='all':
		if is_np:
			res_rotated = np_angles3_rotation(res, rot_angle)
		else:
			f = pytensor.function([], angles3_rotation(res, rot_angle))
			res_rotated = f()
	else:
		raise Exception(f'Not recognized plane {wich_plane}')

	#res_rotated = np_XY_angle_rotation(res, rot_angle)
	#f = pytensor.function([], XY_angle_rotation(res, rot_angle))
	#res_rotated = f()
	print(np.shape(res_rotated))

	ax1 = plt.subplot(2, 4, 1)
	plt.scatter(res[:,0], res[:,1], s=5)
	ax1.set_xlabel('X')
	ax1.set_ylabel('Y')
	if wich_plane in ['XY', 'all']:
		ax1.set_xlim([-90,90])
		ax1.set_ylim([-90,90])

	ax2 = plt.subplot(2, 4, 2)
	plt.scatter(res[:,2], res[:,1], s=5)
	ax2.set_xlabel('Z')
	ax2.set_ylabel('Y')
	if wich_plane in ['YZ', 'all']:
		ax2.set_xlim([10, 130])
		if centered:
			ax2.set_xlim([-30, 30])
		ax2.set_ylim([-120,120])

	ax3 = plt.subplot(2, 4, 5)
	plt.scatter(res[:,0], res[:,2], s=5)
	ax3.set_xlabel('X')
	ax3.set_ylabel('Z')
	if wich_plane in ['XZ', 'all']:
		ax3.set_xlim([-90, 90])
		ax3.set_ylim([30,130])
		if centered:
			ax3.set_ylim([-30,30])

	ax4 = plt.subplot(2, 4, 3)
	plt.scatter(res_rotated[:,0], res_rotated[:,1], s=5)
	ax4.set_xlabel('X')
	ax4.set_ylabel('Y')
	if wich_plane in ['XY', 'all']:
		ax4.set_xlim([-90,90])
		ax4.set_ylim([-90,90])

	ax5 = plt.subplot(2, 4, 4)
	plt.scatter(res_rotated[:,2], res_rotated[:,1], s=5)
	ax5.set_xlabel('Z')
	ax5.set_ylabel('Y')
	if wich_plane in ['YZ', 'all']:
		ax5.set_xlim([10, 130])
		if centered:
			ax5.set_xlim([-30, 30])
		ax5.set_ylim([-120,120])

	ax6 = plt.subplot(2, 4, 7)
	plt.scatter(res_rotated[:,0], res_rotated[:,2], s=5)
	ax6.set_xlabel('X')
	ax6.set_ylabel('Z')
	if wich_plane in ['XZ', 'all']:
		ax6.set_xlim([-90, 90])
		ax6.set_ylim([30,130])
		if centered:
			ax6.set_ylim([-30,30])
	plt.show()



#=========================== Test ============================================================

def show_dist_quaternions(size_val, verbose=True):
	import pandas as pd
	import matplotlib.pyplot as plt
	a = np.random.random(size=(3))
	a = a / np.linalg.norm(a)
	print(a)
	perezsala_parameters = np.random.uniform(size=(size_val,3))
	rotated_list = []
	for i in range(size_val):
		rotated = np_random_uniform_rotation_cluster_to_galactic(a, perezsala_parameters[i,:])
		rotated_list.append(rotated.T)
	df = pd.DataFrame(rotated_list, columns=['x', 'y', 'z'])
	count_x, _ = np.histogram(df.get('x'), bins=8)
	mean_x = np.mean(count_x)
	count_y, _ = np.histogram(df.get('y'), bins=8)
	mean_y = np.mean(count_y)
	count_z, _ = np.histogram(df.get('z'), bins=8)
	mean_z = np.mean(count_z)
	fig = plt.figure()
	ax = plt.subplot(2,2,1)
	df.get('x').hist(bins=8)
	plt.axhline(mean_x, color='r')
	plt.title('X dist')
	ax = plt.subplot(2,2,2)
	df.get('y').hist(bins=8)
	plt.axhline(mean_y, color='r')
	plt.title('Y dist')
	ax = plt.subplot(2,2,3)
	df.get('z').hist(bins=8)
	plt.axhline(mean_z, color='r')
	plt.title('Z dist')
	if verbose:
		plt.show()
	print(f'X mean: {mean_x}, X std: {np.std(count_x)}\n')
	print(f'Y mean: {mean_y}, Y std: {np.std(count_y)}\n')
	print(f'Z mean: {mean_z}, Z std: {np.std(count_z)}\n')

def test_unif_rotation(size_val, confidence):
	from scipy import stats
	a = np.random.random(size=(3))
	a = a / np.linalg.norm(a)
	print("======== Testing Uniform Random Rotation ===============")
	perezsala_parameters = np.random.uniform(size=(size_val,3))
	rotated_list = []
	for i in range(size_val):
		rotated = np_random_uniform_rotation_cluster_to_galactic(a, perezsala_parameters[i,:])
		rotated_list.append(rotated.T)
	rotated_list = np.array(rotated_list)
	print("---------------------- X Coord -------------------------")
	# count_x, _ = np.histogram(rotated_list[:,0], bins=8)
	# mean_x = np.mean(count_x)
	# std_x = np.std(count_x)
	#np.testing.assert_(std_x < (mean_x * 0.06), msg='Fail at X coord')

	ks_test_x = stats.kstest(rotated_list[:,0], stats.uniform(loc=-1, scale=2).cdf)
	np.testing.assert_(ks_test_x.pvalue > (1 - confidence), msg='Fail at X coord')
	print("                          OK                            ")
	print("--------------------------------------------------------")

	print("---------------------- Y Coord -------------------------")
	# count_y, _ = np.histogram(rotated_list[:,1], bins=8)
	# mean_y = np.mean(count_y)
	# std_y = np.std(count_y)
	# np.testing.assert_(std_y < (mean_y * 0.06), msg='Fail at Y coord')

	ks_test_y = stats.kstest(rotated_list[:,1], stats.uniform(loc=-1, scale=2).cdf)
	np.testing.assert_(ks_test_y.pvalue > (1 - confidence), msg='Fail at Y coord')
	print("                          OK                            ")
	print("--------------------------------------------------------")

	print("---------------------- Z Coord -------------------------")
	# count_z, _ = np.histogram(rotated_list[:,2], bins=8)
	# mean_z = np.mean(count_z)
	# std_z = np.std(count_z)
	# np.testing.assert_(std_z < (mean_z * 0.06), msg='Fail at Z coord')

	ks_test_z = stats.kstest(rotated_list[:,2], stats.uniform(loc=-1, scale=2).cdf)
	np.testing.assert_(ks_test_z.pvalue > (1 - confidence), msg='Fail at Y coord')
	print("                          OK                            ")
	print("--------------------------------------------------------")

	print("========================================================")
	
def test_rotaton_norm(size_val):
	a = np.random.random(size=(3))
	norm_a = np.linalg.norm(a)
	print("==== Testing Same Norm Vector after Rotation ===========")
	perezsala_parameters = np.random.uniform(size=(size_val,3))
	rotated_list = []
	for i in range(size_val):
		rotated = np_random_uniform_rotation_cluster_to_galactic(a, perezsala_parameters[i,:])
		rotated_list.append(rotated.T)
	rotated_list = np.array(rotated_list)
	norm_of_rotated = np.linalg.norm(rotated_list, axis=1)
	np.testing.assert_allclose(norm_of_rotated.sum(), size_val*norm_a, rtol=1e-5, atol=0, err_msg='Vectors produced by Uniform Random Rotation are not Unitary')
	print("--------------------------------------------------------")
	print("                          OK                            ")
	print("--------------------------------------------------------")

	print("========================================================")
	

def test_unif_tt(verbose=True):
	print("== Testing Uniform Random Rotation with pytensor =========")
	a = np.random.random(size=(3))
	a = a / np.linalg.norm(a)
	if verbose:
		print('A: ',a)
	perezsala_parameters = np.random.uniform(size=(3))
	rotated = np_random_uniform_rotation_cluster_to_galactic(a, perezsala_parameters)
	if verbose:
		print('Rotated: ',rotated)
	
	a_tt = tt.vector()
	perezsala_parameters_tt = tt.vector()
	f = pytensor.function([a_tt, perezsala_parameters_tt], random_uniform_rotation_cluster_to_galactic(a_tt, perezsala_parameters_tt))
	rotated_tt = f(a, perezsala_parameters)
	if verbose:
		print('Rotated tt: ',rotated_tt)

	np.testing.assert_allclose(rotated, rotated_tt, rtol=1e-10, atol=0, err_msg='Not the same vector')
	print("--------------------------------------------------------")
	print("                          OK                            ")
	print("--------------------------------------------------------")

	print("========================================================")
	
def test_correct_conversion_to_euler_angles_np(verbose=False):
	print("==== Testing PerezSala Params to Euler Angles ==========")
	perezsala_parameters = np.random.random((3))
	if verbose:
		print(perezsala_parameters)
	
	rot_mat = np_random_uniform_rotation_cluster_to_galactic([], perezsala_parameters, False)
	rot_mat = np.array(rot_mat)

	from scipy.spatial.transform import Rotation
	r =  Rotation.from_matrix(rot_mat)
	angles_scipy = r.as_euler("xyz", degrees=False)

	angles_direct = np_perezsala_to_eulerangles(perezsala_parameters)
	
	if verbose:
		print('Rotation_matrix: ',rot_mat)
		print('Angles with scipy: ', angles_scipy)
		print('Angles with Quaternions direct conversion: ', angles_direct)
	
	np.testing.assert_allclose(angles_scipy, angles_direct, rtol=1e-10, atol=0, err_msg='Not the same conversion to euler angle')
	print("--------------------------------------------------------")
	print("                          OK                            ")
	print("--------------------------------------------------------")

	print("========================================================")

def test_conversion_to_euler_tt(verbose=False):
	print("== Testing PerezSala to Euler Angles with pytensor =======")
	perezsala_parameters = np.random.uniform(size=(3))
	eulerangles = np_perezsala_to_eulerangles(perezsala_parameters)
	if verbose:
		print('Euler angles [rad]: ', eulerangles)
	
	perezsala_parameters_tt = tt.vector()
	f = pytensor.function([perezsala_parameters_tt], perezsala_to_eulerangles(perezsala_parameters_tt))
	eulerangles_tt = f(perezsala_parameters)
	if verbose:
		print('Euler angles [rad] tt: ', eulerangles_tt)

	np.testing.assert_allclose(eulerangles, eulerangles_tt, rtol=1e-10, atol=0, err_msg='Not the same conversion to euler angle')
	print("--------------------------------------------------------")
	print("                          OK                            ")
	print("--------------------------------------------------------")

	print("========================================================")

def prueba_rotacion_3():
	xyz = np.array([1,1,0])
	loc = np.array([10,0,0])#np.zeros(3)
	rot_angles = np.array([-np.pi/2, 0, 0])

	f = pytensor.function([],cluster_to_galactic_3ang(xyz, rot_angles, loc))
	xyz_rotated = f()
	print(xyz_rotated)


if __name__ == "__main__":
	stars = np.array([
		               [68.98016279,16.50930235,48.94,63.45,-188.94,54.398],   # Aldebaran
		               [297.69582730,+08.86832120,194.95,536.23,385.29,-26.60] # Altair
		               ])
	#test_Rotation(stars)
	#test_3D(stars)
	#test_6D(stars)
	#rotated = np_random_uniform_rotation_cluster_to_galactic(stars[:,:3].T, np.array([0.2, 0.1, 0.1]))
	#print('Sol:',rotated[0])
	#print('Q: ',np.array(rotated[1]))
	# show_dist_quaternions(8*600, verbose=True)
	# test_unif_rotation(8*600, 0.90)
	# test_rotaton_norm(8*600)
	# test_unif_tt(False)
	# test_correct_conversion_to_euler_angles_np()
	# test_conversion_to_euler_tt()

	#angles = np_perezsala_to_eulerangles([0.502, 0.417, 0.403])
	#print(angles)
	#print(np.rad2deg(angles))
	#found_angley()
	#search_besty_rotation()
	#compare_conversions()
	apply_compare_rotation('all')
	#prueba_rotacion_3()

