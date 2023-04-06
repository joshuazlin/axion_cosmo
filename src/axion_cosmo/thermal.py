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
