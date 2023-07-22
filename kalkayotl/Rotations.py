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

def np_random_uniform_rotation_cluster_to_galactic(xyz, perezsala_parameters, is_dot=True):
	theta1 = 2*np.pi*perezsala_parameters[1]
	theta2 = 2*np.pi*perezsala_parameters[2]
	r1 = np.sqrt(1 - perezsala_parameters[0])
	r2 = np.sqrt(perezsala_parameters[0])
	q = np_quaternions_rotation_matrix(np.cos(theta2)*r2, np.sin(theta1)*r1, np.cos(theta1)*r1, np.sin(theta2)*r2)

	res = q
	if is_dot:
		res = np.dot(xyz, q)
	
	return res

def np_translation_cluster_to_galactic_by_matrix(loc_galactic, tam=4):
    eye = np.eye(tam)
    loc_galactic = np.append(loc_galactic, 1)
    eye[:,tam-1] = loc_galactic
    return eye

def np_translation_cluster_to_galactic(perezsala_parameters, loc_galactic):
	return perezsala_parameters + loc_galactic

def np_cluster_to_galactic_by_matrix(xyz, perezsala_parameters, loc_galactic):
    q = np_random_uniform_rotation_cluster_to_galactic(xyz, perezsala_parameters, is_dot=False)
    t = np_translation_cluster_to_galactic(loc_galactic)
    rotated = np.dot(q, xyz)
    return np.dot(t, np.append(rotated, 1))[:-1]

def np_cluster_to_galactic(xyz, perezsala_parameters, loc_galactic):
    q = np_random_uniform_rotation_cluster_to_galactic(xyz, perezsala_parameters, is_dot=False)
    rotated = np.dot(xyz, q)
    return np_translation_cluster_to_galactic(rotated, loc_galactic)


def quaternions_rotation_matrix(a,b,c,d):
	r = tt.zeros((3,3))

	r_0 = [1 - 2*(c**2 + d**2), 2*(b*c - a*d), 2*(b*d + a*c)]
	r_1 = [2*(b*c + a*d), 1 - 2*(b**2 + d**2), 2*(c*d - a*b)]
	r_2 = [2*(b*d - a*c), 2*(c*d + a*b), 1 - 2*(b**2 + c**2)]

	r = tt.set_subtensor(r[0],r_0)
	r = tt.set_subtensor(r[1],r_1)
	r = tt.set_subtensor(r[2],r_2)

	return r

def random_uniform_rotation_cluster_to_galactic(xyz, perezsala_parameters):
	theta1 = 2*np.pi*perezsala_parameters[1]
	theta2 = 2*np.pi*perezsala_parameters[2]
	r1 = tt.sqrt(1 - perezsala_parameters[0])
	r2 = tt.sqrt(perezsala_parameters[0])
	q = quaternions_rotation_matrix(tt.cos(theta2)*r2, tt.sin(theta1)*r1, tt.cos(theta1)*r1, tt.sin(theta2)*r2)
	
	return tt.dot(xyz, q)

def translation_cluster_to_galactic(xyz, loc_galactic):
	return xyz + loc_galactic

def cluster_to_galactic(xyz, perezsala_parameters, loc_galactic):
    rotated = random_uniform_rotation_cluster_to_galactic(xyz, perezsala_parameters)
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
	print("== Testing Uniform Random Rotation with Theano =========")
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
	f = theano.function([a_tt, perezsala_parameters_tt], random_uniform_rotation_cluster_to_galactic(a_tt, perezsala_parameters_tt))
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
	print("== Testing PerezSala to Euler Angles with Theano =======")
	perezsala_parameters = np.random.uniform(size=(3))
	eulerangles = np_perezsala_to_eulerangles(perezsala_parameters)
	if verbose:
		print('Euler angles [rad]: ', eulerangles)
	
	perezsala_parameters_tt = tt.vector()
	f = theano.function([perezsala_parameters_tt], perezsala_to_eulerangles(perezsala_parameters_tt))
	eulerangles_tt = f(perezsala_parameters)
	if verbose:
		print('Euler angles [rad] tt: ', eulerangles_tt)

	np.testing.assert_allclose(eulerangles, eulerangles_tt, rtol=1e-10, atol=0, err_msg='Not the same conversion to euler angle')
	print("--------------------------------------------------------")
	print("                          OK                            ")
	print("--------------------------------------------------------")

	print("========================================================")


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

	angles = np_perezsala_to_eulerangles([0.502, 0.417, 0.403])
	print(angles)
	print(np.rad2deg(angles))

