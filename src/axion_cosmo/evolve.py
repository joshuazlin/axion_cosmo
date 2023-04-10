"""
def evolve():
    Input: 2 x N x N (x N)
    Output: 2 x N x N (x N) (xT)
"""

from .evolve_utils import *
import pickle
import time
import tqdm
import h5py

def evolve_PQ(shape,
              fa, 
              init_field, 
              init_fieldp, 
              etaini,
              deta, 
              Nstep,
              name=None,
              logdir=None,
              tolog=[],
              flush=100,
              debug=False):
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
               tolog : list containing [("name",lambda f_fp: ..., how often to log this thing, shape of the thing, dtype),...]
               flush : how often to flush the logs
               debug : whether to print out semi-useless debug statements
    """

    R1,T1,t1 = init_params(fa,81)           #gstar fixed to 81 for now
    field    = init_field(shape)            #Init field
    fieldp   = init_fieldp(shape)           #and its derivative
    y_yp     = np.vstack((field,fieldp))    #Stackem
    eta      = etaini                       #initial eta
    
    for i in tqdm.tqdm(range(Nstep)):
        if debug:
            print("running",i,np.average(np.abs(y_yp)))
            time.sleep(0.1)

        if i%flush == 0:
            logfile = h5py.File(f'{logdir}/{name}_{i//flush}.hdf5', 'w')
            datasets = []
                for x in tolog:
                    datasets.append(logfile.create_dataset(x[0], (flush//x[2]+1,) + x[3], x[4]))

        for j,x in enumerate(tolog):
            if i%x[2] == 0:
                datasets[j][i//x[2]] = x[1](y_yp)

        if i%flush == flush - 1:
            logfile.flush()
            logfile.close()

        y_yp = RK4(lambda eta,y_yp : np.vstack((y_yp[2:],PQ_epoch_diff(eta,y_yp[:2],y_yp[2:],R1,T1,t1,fa,81,debug=debug))),
                   eta,y_yp,deta)
        eta += deta

    return 
