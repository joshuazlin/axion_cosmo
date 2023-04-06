"""
def evolve():
    Input: 2 x N x N (x N)
    Output: 2 x N x N (x N) (xT)
"""

from .evolve_utils import *
import pickle

def evolve_PQ(shape, fa, init_field, init_fieldp, t_final, name=None, logdir=None):
    """
    Evolution in the PQ epoch

    Inputs
        init: 2 x N x N (x N) field representing initial conditions for equations (S9) and (S10)
        dt: timestep
        N: number of steps to take
        logdir: (Optional) directory where to write field configurations, every logstep
                pass it as an absolute or suffer the consequences

    fa ~ 2.27e14 MeV
    """

    R1,T1 = init_params(fa)
    field = init_field(shape)
    fieldp = init_fieldp(shape)

    ev = scipy.integrate.RK45(lambda t,y_yp : np.vstack((y_yp[2:],PQ_epoch_diff(t,y_yp[:2],y_yp[2:],R1,T1,fa)),
                         0.0001,np.vstack((field,fieldp))),800)
    
    i = 0
    while ev.step():
        if i%logstep == 0 and logdir is not None:
            pickle.dump(ev.y[:2],open(f"{logdir}/{name}_{i}")
        i++

    













