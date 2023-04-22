"""
def evolve():
    Input: 2 x N x N (x N)
    Output: 2 x N x N (x N) (xT)
"""

from .evolve_utils import *
from .params import *
from .thermal import *
import pickle
import time
import tqdm
import h5py

def evolve(shape,
              params,
              field=None,
              fieldp=None,
              name=None,
              logdir=None,
              tolog=[],
              flush=100,
              debug=False):
    """
    Evolution in the PQ epoch

    Inputs
               shape : shape of spatial dims
                  fa : in units of m_planck
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

    params = update_params(params)
    print(f'CHECK: Simulation from eta = {params["eta_PQ"]} to eta = {params["eta_PQ"] + params["Nstep"]*params["deta"]}')

    if field is None:
        y_yp = thermal(shape,params)
        y_yp = np.array([y_yp[0] + 1j*y_yp[1],y_yp[2] + 1j*y_yp[3]])
    else:
        y_yp = np.vstack((field,fieldp))    #Stackem

    for i in tqdm.tqdm(range(params['Nstep'])):
        if debug:
            print("running",i,np.average(np.abs(y_yp)))
            time.sleep(1)

        if i%flush == 0:
            #Final Log
            #if i != 0:
            #    for j,x in enumerate(tolog):
            #        if (i+1)%x[2] == 0:
            #            datasets[j][((i+1)%flush)//x[2]] = x[1](y_yp)
            #    logfile.flush()
            #    logfile.close()
            if i != 0:
                logfile.flush()
                logfile.close()

            logfile = h5py.File(f'{logdir}{name}_{i//flush}.hdf5', 'w')
            datasets = []
            for x in tolog:
                datasets.append(logfile.create_dataset(x[0], (flush//x[2],) + x[3], x[4]))
                for k in params.keys():
                    datasets[-1].attrs[k] = params[k]

        if params['stage'] == 'PQ':
            #y_yp = RK4(lambda eta,y_yp : \
            #    np.vstack((y_yp[2:],PQ_epoch_diff(y_yp[:2],y_yp[2:],params,debug=debug))),
            #           params['eta_PQ'],y_yp,params['deta'])
            y_yp = RKN4(lambda eta, y: PQ_epoch_diff_rescaled(y,params,debug=debug),
                        params['eta_PQ'],y_yp[0],y_yp[1],params['deta'])
            params['eta_PQ'] = params['eta_PQ'] + params['deta']
        elif params['stage'] == 'earlyQCD':
            y_yp = RK4(lambda eta,y_yp : \
                np.vstack((y_yp[2:],earlyQCD_epoch_diff(y_yp[:2],y_yp[2:],params,debug=debug))),
                       params['eta_QCD'],y_yp,params['deta'])
            params['eta_QCD'] = params['eta_QCD'] + params['deta']
        else:
            raise NotImplemented

        for j,x in enumerate(tolog):
            if (i+1)%x[2] == 0:
                datasets[j][(i%flush)//x[2]] = x[1](y_yp)

    #for j,x in enumerate(tolog):
    #    if (params['Nstep'])%x[2] == 0:
    #        datasets[j][(params['Nstep']%flush)//x[2]] = x[1](y_yp)
    logfile.flush()
    logfile.close()

    return
