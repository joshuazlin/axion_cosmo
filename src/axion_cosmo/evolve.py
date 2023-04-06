"""
def evolve():
    Input: 2 x N x N (x N)
    Output: 2 x N x N (x N) (xT)
"""

from .evolve_utils import *
import pickle
import time

def evolve_PQ(shape,
              fa, 
              init_field, 
              init_fieldp, 
              etaini,
              deta, 
              Nstep,
              name=None, logdir=None, tolog=None, flush=None, debug=False):
    """
    Evolution in the PQ epoch

    Inputs
               shape : shape of spatial dims
                  fa : axion decay constant ~ 2.27e14 MeV
          init_field : fn that takes shape and spits out an initial config
          init_field : fn that takes shape and spits out an initial config'
              etaini : starting eta
                deta : d_eta
               Nstep : how many steps
                name : name of logfile
              logdir : where to save logfile
               tolog : list containing [("name",lambda f,fp: ..., how often to log this thing),...]
               flush : how often to flush the logs
               debug : whether to print out semi-useless debug statements
    """

    R1,T1,t1 = init_params(fa,81) #gstar fixed to 81 for now
    field    = init_field(shape)
    fieldp   = init_fieldp(shape)
    y_yp     = np.vstack((field,fieldp))
    eta      = etaini
    logs     = {}
    for x in tolog:
        #print("huh?",x,logs,"huh??")
        logs[x[0]] = []

    for i in range(Nstep):
        if debug:
            print("runnin",i,np.average(np.abs(y_yp)))
            time.sleep(0.1)

        if tolog is not None:
            assert name is not None
            assert logdir is not None
            for x in tolog:
                #print(x)
                if i%x[2] == 0:
                    logs[x[0]].append(x[1](y_yp[:2],y_yp[2:]))
            if flush is not None:
                if i%flush == 0:
                    pickle.dump(logs,open(f"{logdir}/{name}","wb"))
        y_yp = RK4(lambda t,y_yp : np.vstack((y_yp[2:],PQ_epoch_diff(eta,y_yp[:2],y_yp[2:],R1,T1,t1,fa,81,debug=debug))),
                   eta,y_yp,deta)
        eta += deta
    pickle.dump(logs,open(f"{logdir}/{name}","wb"))
    return logs











