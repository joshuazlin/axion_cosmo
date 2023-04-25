
import axion_cosmo as ac
import numpy as np
import sys
import h5py

# first run PQ, then use the same file to run QCD

# forcing logdir is current directory
logdir=''

# command line input of shape 
shape = tuple(map(int, sys.argv[1:]))
print("spatial dimensions: ",shape)

# first run PQ
params={
  'fa' : 1e-3,
  'alpha' : 1.68e-7,
  'n' : 6.68,
  'Lambda' : 1e-4,
  'eta_PQ' : 1e-2,
  'eta_QCD' : 0.1,
  'deta' : 1e-3,
  'Nstep' : 10000,
  'a' : 1,
  'stage' : 'PQ',
  'kmax' : 4,
  'gstar' : 81,
  'set_lambda_tilde' : 100,
  'set_etac_QCD' : 2,
  'set_lambda' : 0.1,
  'set_scale' : 0.5,
  'set_c' : 1,
  'Ntherm' : 1,
}
ac.evolve(shape=shape,
          params=params,
          field=None,
          fieldp=None,
          name=f'PQ_ev_{shape}',
          logdir=logdir,
          tolog=[("field",lambda f_fp:f_fp,50,(2,)+shape,np.complex64),
                 ("vev",lambda f_fp:np.average(np.abs(f_fp[0])),1,(1,),np.float64)],
          flush=params['Nstep'],)

# now run QCD
f = h5py.File(f'{logdir}PQ_ev_{shape}_0.hdf5', 'r')

params['stage'] = 'earlyQCD'  # change stage
params['set_scale'] = 1  # change scale of physical vs lattice box
params['set_lambda_tilde'] = 1  # just to make field value converge to vev faster
params['deta'] = 5e-4  # change time step size so QCD evolution runs more precisely


ac.evolve(shape=shape,
          params=params,
          field=f['field'][-1,0],  # initial condition for QCD evolution is the last step of PQ evolution
          fieldp=f['field'][-1,1],
          name=f'QCD_ev_{shape}',
          logdir=logdir,
          tolog=[("field",lambda f_fp:f_fp,50,(2,)+shape,np.complex64),
                 ("vev",lambda f_fp:np.average(np.abs(f_fp[0])),1,(1,),np.float64)],
          flush=params['Nstep'],)

print("Finished!")




