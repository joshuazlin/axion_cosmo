"""Utils for evolution

Utility functions for evolution. Mostly everything has been written by hand. 
TODO: write things in jax, gpu support, mpi4jax to split up array? Should be not too hard
"""
from .utils import *
import numpy as np
import scipy

def RK4(f,t,y,h):
    """One RK4 step, assumes f(t,y) := y', with a stepsize h
    """
    k1 = f(t,y)
    k2 = f(t+h/2,y+h*k1/2)
    k3 = f(t+h/2,y+h*k2/2)
    k4 = f(t+h,y+h*k3)
    return y + (1/6)*(k1+2*k2+2*k3+k4)*h

def RKN4(f,t,y,yp,h):
    """One RKN step, assumes f(t,y) := y'', with a stepsize h
    """
    k1 = f(t,y)
    k2 = f(t+ (h/2),y + (h/2)*yp + (h**2)*(1/8)*k1)
    k3 = f(t+h, y+h*yp+(h**2)*(1/2)*k2)
    return np.array([y + h*yp + (h**2)*((k1/6)+(k2/3)),
            yp + h*((k1/6)+(4*k2/6)+(k3/6))])

def grad(field,dir,order,N):
    """Gradient of a field in some direction at some order.

    Take derivative of a field, in a specific direction,
    with some number of points included (stencil number).
    TODO: think of it as a convolved stencil, use fft methods
    (fix the fourier transform)

    Inputs:
        field: real numpy array
        dir: axis to compute
        order: order of derivative to compute
        N: number of points to include
    """

    if (order,N) == (2,3):
        return (np.roll(field,-1,axis=dir) + \
               -2*field + \
               np.roll(field,1,axis=dir))
    if (order,N) == (2,7):
        return (2*np.roll(field,-3,axis=dir) + \
               -27*np.roll(field,-2,axis=dir) + \
               270*np.roll(field,-1,axis=dir) + \
               -490*field + \
               270*np.roll(field,1,axis=dir) + \
               -27*np.roll(field,2,axis=dir) + \
               2*np.roll(field,3,axis=dir))/180
    else:
        print(order,N)
        raise NotImplementedError

def rescaled_nabla(field,a,N=3):
    """Rescaled laplacian of a field

    Inputs:
        field: complex field, possibly 2D, or 3D, with shape
               Lx x Ly x (Lz)
            a: scale factor

    Returns:
      Rescaled laplacian (by scale factor a)
    """
    return np.sum([grad(field,i,2,N) for i in range(len(field.shape))],axis=0)/(a**2)

def PQ_epoch_diff_rescaled(field,params,lamb=1,debug=False):
    """Returns the second derivative of the field in PQ era

    Returns the second derivative of the field, when we have rewritten the
    equation to not have single derivative terms. See the writeup for details.

    Inputs:
       field: complex field, possibly 2D or 3D, with shape
              Lx x Ly x (Lz)
      params: dictionary containing cosmological parameters.
        lamb: lambda, set to one by default.

    Returns:
      Second derivative of field (with respect to eta)
    """
    return rescaled_nabla(field,params['a']/params['fa']) \
            -lamb*field*(np.abs(field)**2 - params['eta_PQ']**2 + params['c'])

def earlyQCD_epoch_diff_rescaled(field,params,debug=False):
    """Returns the second derivative of the field in earlyQCD era.

    Returns the second derivative of the field, when we have rewritten the
    equation to not have single derivative terms. See the writeup for details.

    Inputs:
       field: complex field, possibly 2D or 3D, with shape
              Lx x Ly x (Lz)
      params: dictionary containing cosmological parameters.

    Returns:
      Second derivative of field (with respect to eta)
    """
    return rescaled_nabla(field,params['a']/params['fa']) \
           -params['set_lambda_tilde']*field*(np.abs(field)**2 - params['eta_QCD']**2) \
           -params['eta_QCD']**4*min(params['eta_QCD'],params['set_etac_QCD'])**params['n']*np.abs(field)**(-3)*\
            (-np.imag(field)**2+1j*np.real(field)*np.imag(field))

def PQ_epoch_diff(field,fieldp,params,lamb=1,debug=False):
    """DEPRECATED: please see PQ_epoch_diff_rescaled

    y'' = f(t,y,y'), this is that function f. (in the PQ epoch)
    a:  dimensionless "Lattice spacing" (old: L_phys (in units of H1) / L_lat (number of sites))
    c : T1**2/3fa**2
    """
    raise Deprecated
    A = -(2/params['eta_PQ'])*fieldp
    B = rescaled_nabla(field,params['a']/params['fa'])
    C = -lamb*field*((params['eta_PQ']**2)*\
                         (np.repeat(np.expand_dims(np.sum(field**2,axis=0),0),2,0) - 1) + params['c'])

    return A + B + C

def earlyQCD_epoch_diff(field,fieldp,params,lamb=1,debug=False):
    """DEPRECATED: please see earlyQCD_epoch_diff_rescaled

    Note that lamb = 1 is the lamb from PQ evolution.
    """
    raise Deprecated
    A = -(2/params['eta_QCD'])*fieldp
    B = rescaled_nabla(field,params['a']*params['eta_PQ']/params['H1_QCD']) #Is this eta_PQ or eta_QCD? Its.... I need to fix this
                                                    #its with respect to a1 H1 x, where 1 is now 1_QCD
    C = -(lamb*params['fa']**2/(params['H1_QCD']**2))\
            *field*(params['eta_QCD']**2)*(np.repeat(np.expand_dims(np.sum(field**2,axis=0),0),2,0) - 1)
    D0 = (min(params['eta_QCD'],params['etac_QCD'])**params['n'])*\
                    (params['eta_QCD']**2)*(np.sum(field**2,axis=0)**(-3/2))
    D1 = -field[1]**2; D2 = field[0]*field[1]
    D = np.stack((D1/D0,D2/D0),axis=0)
    return A + B + C + D

