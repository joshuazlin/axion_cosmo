"""
def evolve():
    Input: 2 x N x N (x N)
    Output: 2 x N x N (x N) (xT)
"""

from .evolve_utils import *
import pickle
import time

def evolve_PQ(shape, fa, init_field, init_fieldp, N, name=None, logdir=None, logstep=None):
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

    R1,T1,t1 = init_params(fa,81)
    field = init_field(shape)
    fieldp = init_fieldp(shape)

    #Why does scipy require 1d? is it the adaptive stuff? yuck
    #ev = scipy.integrate.RK45(lambda t,y_yp : np.vstack((y_yp[2:],PQ_epoch_diff(t,y_yp[:2],y_yp[2:],R1,T1,fa,lambda T : 81))),
    #                     0.0001,np.vstack((field,fieldp)),800)
    
    #i = 0
    #while ev.step():
    #    if i%logstep == 0 and logdir is not None:
    #        pickle.dump(ev.y[:2],open(f"{logdir}/{name}_{i}","wb"))
    #    i += 1
    #pickle.dump(ev.y[:2],open(f"{logdir}/{name}_final","wb"))

    y_yp = np.vstack((field,fieldp))
    for i,t in enumerate(np.linspace(0.0001,800,N)):
        print(i,y_yp)
        time.sleep(3)
        if logdir is not None and logstep is not None:
            if i%logstep == 0:
                pickle.dump(y_yp,open(f"{logdir}/{name}_{i}","wb"))
        y_yp = RK4(lambda t,y_yp : np.vstack((y_yp[2:],PQ_epoch_diff(t,y_yp[:2],y_yp[2:],R1,T1,t1,fa,81))),
                   t,y_yp, (800-0.0001)/(N-1))
    pickle.dump(y_yp,open(f"{logdir}/{name}_final","wb"))


