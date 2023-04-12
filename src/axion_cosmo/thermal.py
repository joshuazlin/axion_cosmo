"""

Code to generate initial thermal configuration

def thermal(...):
    Input: ... temperature? N_modes? N...
    Output: 2 x N x N (x N) real numpy array to feed into evolution scripts
"""

import numpy as np

def dummy_thermal(shape,eps=0.1):
    assert type(shape) is tuple
    return 2*eps*np.random.random((2,) + shape) - eps


def stand_dev_phi(k,L,T,fa):
	'''
	wrapper function to compute standard deviation of the field given wave number k
	taking lambda to 1 as mentioned in the paper
	'''
	meff2 = T^2 / 3. - fa^2
	omegak = np.sqrt(k^2 + meff2)
	nk = 1./(np.exp(omegak/T) - 1)

	return nk / omegak * L

def stand_dev_phidot(k,L,T,fa):
	'''
	wrapper function to compute standard deviation of the time derivative of the field given wave number k
	taking lambda to 1 as mentioned in the paper
	'''
	meff2 = T^2 / 3. - fa^2
	omegak = np.sqrt(k^2 + meff2)
	nk = 1./(np.exp(omegak/T) - 1)

	return nk * omegak * L


def thermal(shape,L,T,fa,kmax=50):
	'''
	generate initial field (and first derivative) values with thermal distribution 

	shape: spatial grid size, e.g. 20 * 20 for 2D or 20 * 20 * 20 for 3D
	L: size of simulation box (different for PQ and QCD epochs!)
	T: temperature
	fa: axion decay constant
	kmax: the max number of k-modes summed over in Fourier Transform
	'''
	assert type(shape) is tuple
	dim = len(shape)  # spatial dimensions
	
	# generate the first 50 k-modes in all spatial dims
	phi1_k = np.array([np.random.normal(0.,stand_dev_phi(k_i,L,T,fa),dim) for k_i in range(kmax)])
	phi2_k = np.array([np.random.normal(0.,stand_dev_phi(k_i,L,T,fa),dim) for k_i in range(kmax)])
	phi1_dot_k = np.array([np.random.normal(0.,stand_dev_phidot(k_i,L,T,fa),dim) for k_i in range(kmax)])
	phi2_dot_k = np.array([np.random.normal(0.,stand_dev_phidot(k_i,L,T,fa),dim) for k_i in range(kmax)])

	


