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
    """Generic evolution code

    Generic code to do evolution. Writes to h5py files, with path specified by
    f'{logdir}{name}_{i}' where i increments every $flush steps.

    Inputs
       shape : shape of spatial dims
      params : cosmological parameters
       field : Initial field, or None (will initialize a thermal config)
      fieldp : Initial field, or NOne (will initialize a thermal config)
        name : File to write to
      logdir : where to save logfile
       tolog : list containing [("name",lambda f_fp: ..., how often to log this thing, shape of the thing, dtype),...]
       flush : how often to flush the logs
       debug : whether to print out semi-useless debug statements
    """

    params = update_params(params)
    print(f"Simulating {params['stage']} stage")
    if params['stage'] == 'PQ':
        print(f'CHECK: Simulation from eta = {params["eta_PQ"]} to eta = {params["eta_PQ"] + params["Nstep"]*params["deta"]}')
    elif params['stage'] == 'earlyQCD':
        print(f'CHECK: Simulation from eta = {params["eta_QCD"]} to eta = {params["eta_QCD"] + params["Nstep"]*params["deta"]}')

    if field is None:
        y_yp = params['eta_PQ']*thermal(shape,params)
    else:
        if params['stage'] in ['earlyQCD','earlyQCDN']:
            y_yp = params['eta_QCD']*np.array([field,fieldp])    #Stackem
        else:
            raise

    if params['stage'] == 'PQ' and 'Ntherm' in params.keys():
        for i in tqdm.tqdm(range(params['Ntherm'])):
            y_yp = RKN4(lambda eta, y: PQ_epoch_diff_rescaled(y,params,debug=debug),
                        params['eta_PQ'],y_yp[0],y_yp[1],params['deta'])

    for i in tqdm.tqdm(range(params['Nstep'])):
        if debug:
            print("running",i,np.average(np.abs(y_yp)))
            time.sleep(1)

        if i%flush == 0:
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
            y_yp = RKN4(lambda eta, y: PQ_epoch_diff_rescaled(y,params,debug=debug),
                        params['eta_PQ'],y_yp[0],y_yp[1],params['deta'])
            params['eta_PQ'] = params['eta_PQ'] + params['deta']
        elif params['stage'] == 'earlyQCD':
            y_yp = RKN4(lambda eta, y: earlyQCD_epoch_diff_rescaled(y,params,debug=debug),
                        params['eta_QCD'],y_yp[0],y_yp[1],params['deta'])
            params['eta_QCD'] = params['eta_QCD'] + params['deta']
        elif params['stage'] == 'earlyQCDN':
            y_yp = RKN4(lambda eta, y: earlyQCD_epoch_diff_rescaled_N(y,params,debug=debug),
                        params['eta_QCD'],y_yp[0],y_yp[1],params['deta'])
            params['eta_QCD'] = params['eta_QCD'] + params['deta']
        else:
            raise NotImplemented

        for j,x in enumerate(tolog):
            if (i+1)%x[2] == 0:
                datasets[j][(i%flush)//x[2]] = x[1](y_yp)

    logfile.flush()
    logfile.close()

    return
