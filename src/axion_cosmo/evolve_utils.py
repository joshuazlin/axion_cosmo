"""
Note that everywhere I write dt, I probably mean deta
"""
import numpy as np
import scipy

def RK4(f,t,y,h):
    k1 = f(t,y)
    k2 = f(t+h/2,y+h*k1/2)
    k3 = f(t+h/2,y+h*k2/2)
    k4 = f(t+h,y+h*k3)
    return y + (1/6)*(k1+2*k2+2*k3+k4)*h

def grad(field,dir,order,N):
    """
    take derivative of a field, in a specific direction,
    with some number of points included (stencil number)
    This is a cheap-imitation of something that would use non-axis points
    https://web.media.mit.edu/~crtaylor/calculator.html
    Probably thats what they meant by seven-point stencil though, the (2,3) scheme I wrote below

    Inputs:
        field: real numpy array
        dir: axis to compute
        order: order of derivative to compute
        N: number of points to include
    """

    if (order,N) == (2,3):
        return (np.roll(field,-1,axis=dir) + \
               -2*np.roll(field,-1,axis=dir) + \
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
    """
    take rescaled nabla^2 of a field (defined under (S8))
    scale is "a"

    Inputs:
        field: 2 x N x N (x N)
    """
    n_space = len(field.shape) - 1
    to_return = np.zeros(field.shape)
    for i in range(1,n_space + 1):
        to_return = to_return + grad(field,i,2,N)
    return to_return/(a**2)

def PQ_epoch_diff(eta,field,fieldp,c,a,fa,lamb=1,debug=False):
    """
    (S9) and (S10)
    y'' = f(t,y,y'), this is that function f. (in the PQ epoch)
    a:  dimensionless "Lattice spacing" (old: L_phys (in units of H1) / L_lat (number of sites))
    c : T1**2/3fa**2
    """
    A = -(2/eta)*fieldp
    B = rescaled_nabla(field,a*eta/fa)
    C = -lamb*field*((eta**2)*(np.repeat(np.expand_dims(np.sum(field**2,axis=0),0),2,0) - 1) + c)
    #print(np.sum(A),np.sum(np.abs(B)),np.sum(C))
    return A + B + C

def earlyQCD_epoch_diff(eta,field,fieldp,scale,etac,lamb=5504,n=6.68,debug=False):
    A = -(2/eta)*fieldp
    B = rescaled_nabla(field,scale)
    C = -lamb*field*(eta**2)*(np.repeat(np.expand_dims(np.sum(field**2,axis=0),0),2,0) - 1)
    D0 = (min(eta,etac)**n)*(eta**2)*(np.sum(field**2,axis=0)**(-3/2))
    D1 = -field[1]**2; D2 = field[0]*field[1]
    D = np.vstack((D1/D0,D2/D0))
    return A + B + C + D

def lateQCD_epoch_diff():
    pass
