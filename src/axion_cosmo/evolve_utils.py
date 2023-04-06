"""
Note that everywhere I write dt, I probably mean deta
"""
import numpy as np

def grad(field,dir,order,N)
    """
    take derivative of a field, in a specific direction,
    with some number of points included (stencil number)
    This is a cheap-imitation of something that would use non-axis points 

    Inputs:
        field: real numpy array
        dir: axis to compute
        order: order of derivative to compute
        N: number of points to include
    """

    if (order,N) == 7:
        return (2*np.roll(field,-3,axis=dir) + \
               -27*np.roll(field,-2,axis=dir) + \
               270*np.roll(field,-1,axis=dir) + \
               -490*field + \
               270*np.roll(field,1,axis=dir) + \
               -27*np.roll(field,2,axis=dir) + \
               2*np.roll(field,3,axis=dir))/180
    else:
        raise NotImplementedError

def rescaled_nabla(field,scale,N=7):
    """
    take rescaled nabla^2 of a field (defined under (S8))
    It's taking nabla with respect to space "at the beginning", really we just need some scale factor, scale
    
    Inputs:
        field: 2 x N x N (x N)
    """
    n_space = len(field.shape) - 1
    to_return = np.zeros(field.shape)
    for i in range(1,n_space + 1):
        to_return = to_return + grad(field,i,2,N)
    return to_return/(scale**2)

def R(gstar,T):
    """
    (S46) in MeV. 
    """
    return 3.699*1e-10*(gstar(T)**(-1/3))/T

def H(gstar,T):
    """
    (S44), and https://physics.nist.gov/cgi-bin/cuu/Value?plkmc2gev
    """
    return 1.660*(gstar(T)**(1/2))*(T**2)/1.220890e22


def PQ_epoch_diff(t,field,fieldp,R1,T1,fa,lamb=1):
    """
    (S9) and (S10)
    y'' = f(t,y,y'), this is that function f. (in the PQ epoch)
    """
    return -(2/t)*fieldp + rescaled_nabla(field,R(gstar,T)) - lamb*field*((t**2)*(np.repeat(np.expand_dims(np.sum(field**2,axis=0),0),2,0) - 1) + (T1**2)/(3*fa**2))


def init_params(fa,gstar1):
    """

    """
    T1 = scipy.optimize.fsolve(lambda T: H(gstar1,T) - fa, 1)
    R1 = R(gstar1,T)
    return R1,T1

