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


'''from params'''
def R(gstar,T):
    """
    (S46) in MeV. 
    """
    return 3.699*1e-10*(gstar**(-1/3))/T

def H(gstar,T):
    """
    (S44), and https://physics.nist.gov/cgi-bin/cuu/Value?plkmc2gev
    """
    return 1.660*(gstar**(1/2))*(T**2)/1.220890e22

def t(eta,t1):
    """
    (S6)
    """
    return (eta**2)*(t1)

def T(gstar,t):
    """
    S(45)
    """
    return (0.3012*(gstar**(-1/2))*1.220890e22/t)**(1/2)


def init_PQ_params(fa,gstar):
    """
    these ARENT where you initialize, these are at eta = 1
    """
    T1 = (fa*1.220890e22/(1.660*(gstar**(1/2))))**(1/2)
    R1 = R(gstar,T1)
    t1 = t_T(gstar,T1)
    Tc = 1.68e-7*((400))
    return R1,T1,t1
''''''


def stand_dev_phi(ksq,V,T,fa):
	'''
	wrapper function to compute standard deviation of the field given wave number k
	taking lambda to 1 as mentioned in the paper
	'''

	meff2 = T^2 / 3. - fa^2
	omegak = np.sqrt(ksq + meff2)
	nk = 1./(np.exp(omegak/T) - 1)

	return nk / omegak * V

def stand_dev_phidot(ksq,V,T,fa):
	'''
	wrapper function to compute standard deviation of the time derivative of the field given wave number k
	taking lambda to 1 as mentioned in the paper
	'''
	meff2 = T^2 / 3. - fa^2
	omegak = np.sqrt(ksq + meff2)
	nk = 1./(np.exp(omegak/T) - 1)

	return nk * omegak * V


def thermal(shape,
			eta0, 
			gstar, 
			fa,
			scale,
			kmax=10):
	'''
	generate initial field (and first derivative) values with thermal distribution 

	shape: spatial grid size, e.g. 20 * 20 for 2D or 20 * 20 * 20 for 3D
	eta0: tilde eta at which you want to initialize thermally
	gstar: enters functions of scale factor and temperature and so on
	fa: axion decay constant
	scale: L_physical / L_lattice
	kmax: half of the box size of k-modes summed over in Fourier Transform
	'''
	assert type(shape) is tuple

	R1, T1, t1 = init_PQ_params(fa,gstar)
	T0 = T(gstar,t(eta0, t1))
	H0 = H(gstar,T0)
	R0 = R(gstar,T0)


	dim = len(shape)  # spatial dimensions of lattice
	Lx, Ly, Lz = shape
	Lx_phys = Lx * scale / (R0 * H0)
	Ly_phys = Ly * scale / (R0 * H0)
	Lz_phys = Lz * scale / (R0 * H0)
	V = Lx_phys * Ly_phys * Lz_phys

	# initialize arrays
	phi1_k_lattice = np.zeros(shape)
	phi2_k_lattice = np.zeros(shape)
	phi1dot_k_lattice = np.zeros(shape)
	phi2dot_k_lattice = np.zeros(shape)

	# fill whole array with Gaussian dist
	for x in range(Lx):
		# zero momentum is at index int(Lx/2)
		kx = (x - int(Lx/2)) * scale / (R0 * H0)
		for y in range (Ly):
			ky =  (y - int(Ly/2)) * scale / (R0 * H0)
			for z in range(Lz):
				kz = (z - int(Lz/2)) * scale / (R0 * H0)
				ksq = kx^2 + ky^2 + kz^2
				phi1_k_lattice[x,y,z], phi2_k_lattice[x,y,z] = np.random.normal(0.,stand_dev_phi(ksq,V,T0,fa))
				phi1dot_k_lattice[x,y,z], phi2dot_k_lattice[x,y,z] =np.random.normal(0.,stand_dev_phidot(ksq,V,T0,fa))


	# mask out with zeros
	# wants to keep int(Lx/2) - kmax to int(Lx/2) + kmax
	phi1_k_lattice[:int(Lx/2)-kmax,:,:] = 0
	phi1_k_lattice[int(Lx/2)+kmax:,:,:] = 0
	phi2_k_lattice[:int(Lx/2)-kmax,:,:] = 0
	phi2_k_lattice[int(Lx/2)+kmax:,:,:] = 0
	phi1_k_lattice[:,:int(Ly/2)-kmax,:] = 0
	phi1_k_lattice[:,int(Ly/2)+kmax:,:] = 0
	phi2_k_lattice[:,:int(Ly/2)-kmax,:] = 0
	phi2_k_lattice[:,int(Ly/2)+kmax:,:] = 0	
	phi1_k_lattice[:,:,:int(Lz/2)-kmax] = 0
	phi1_k_lattice[:,:,int(Lz/2)+kmax:] = 0
	phi2_k_lattice[:,:,:int(Lz/2)-kmax] = 0
	phi2_k_lattice[:,:,int(Lz/2)+kmax:] = 0


	# Inverse Fourier Transform!
	phi1_x_lattice = np.fft.irfftn(phi1_k_lattice,shape)
	phi2_x_lattice = np.fft.irfftn(phi2_k_lattice,shape)
	phi1dot_x_lattice = np.fft.irfftn(phi1dot_k_lattice,shape)
	phi2dot_x_lattice = np.fft.irfftn(phi2dot_k_lattice,shape)


	# return psi not phi fields
	# including jacobian factor from dt to d eta
	jac = 2 * sqrt(t(eta0, t1)/ t1)
	return phi1_x_lattice / fa, phi2_x_lattice / fa, jac * phi1dot_x_lattice / fa, jac * phi2dot_x_lattice / fa





	


