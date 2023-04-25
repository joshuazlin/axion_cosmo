"""

Code to generate initial thermal configuration

def thermal(...):
    Input: ... temperature? N_modes? N...
    Output: 2 x N x N (x N) real numpy array to feed into evolution scripts
"""

import numpy as np

def thermal(shape,
            params,
            lamb=1):
    """Initial thermal distribution
    
    Generates an initial thermal distribution, to be used 
    before the PQ transition. 
    
    Inputs:
       shape: shape of the field
      params: cosmological params
      
    Returns
      complex field, along with its derivative in eta. 
      Returns the field *not* scaled by eta (so just in the dimensionless form, 
      where the vev breaks to one.)
    """
    assert type(shape) is tuple
    
    #Easiest to initialize lattice momenta
    k2_lat = np.sum((np.sin(2*np.pi*(np.indices(shape).T/np.array(shape).T).T)**2),axis=0)
    if (params['T']**2)/3 - params['fa']**2 < 0:
        print('ERROR: you have attempted to create a thermal configuration after symmetry breaking!')
        print(f"T:{params['T']},fa/sqrt(3):{params['fa']/np.sqrt(3)}")
        raise
    wk = np.sqrt(k2_lat/((params['a']*params['eta_PQ']/params['fa'])**2) + \
                             lamb*((params['T']**2)/3 - params['fa']**2)) #dimension 1
    nk = 1/(np.exp(wk/params['T']) - 1)
    
    # fill whole array with Gaussian dist
    field   = np.random.normal(0,np.sqrt((nk/wk)*np.prod(shape)*(params['a']*params['eta_PQ']/params['fa'])**(-3)),
                               (2,)+shape)
    field_p = np.random.normal(0,np.sqrt(nk*wk*np.prod(shape)*(params['a']*params['eta_PQ']/params['fa'])**(-3)),
                               (2,)+shape)

    # mask out with zeros
    dist = np.sum([np.min(np.array([x**2,(shape[i]-x)**2]),axis=0) for i,x in enumerate(np.indices(shape))],axis=0)
    field[0]   = np.where(dist < params['kmax']**2, field[0], 0)
    field[1]   = np.where(dist < params['kmax']**2, field[1], 0)
    field_p[0] = np.where(dist < params['kmax']**2, field_p[0], 0)
    field_p[1] = np.where(dist < params['kmax']**2, field_p[1], 0)

    # Inverse Fourier Transform!
    field   = np.fft.irfftn(field,shape)
    field_p = np.fft.irfftn(field_p,shape)

    # return psi not phi fields
    # including jacobian factor from dt to d eta
    return np.array([(field[0] + 1j*field[1])/params['fa'],
                     (field_p[0]+1j*field_p[1])*params['eta_PQ']/(params['fa']**2)])


