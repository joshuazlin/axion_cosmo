
import axion_cosmo as ac
import numpy as np

params={
  'fa' : 1e-3,
  'alpha' : 1.68e-7,
  'n' : 6.68,
  'Lambda' : 1e-4,
  'eta_PQ' : 1e-2,
  'deta' : 1e-3,
  'Nstep' : 10000,
  'a' : 1,
  'stage' : 'PQ',
  'kmax' : 4,
  'gstar' : 81,
  'set_lambda' : 0.1,
  'set_scale' : 0.5,
  'set_c' : 1,
  'Ntherm' : 1000,
}
evolve(shape=(100,100,100),
          params=params,
          field=None,
          fieldp=None,
          name='PQ_ev',
          logdir='',
          tolog=[("field",lambda f_fp:f_fp,50,(2,100,100,100),np.complex64),
                 ("vev",lambda f_fp:np.average(np.abs(f_fp[0])),1,(1,),np.float64)],
         flush=10000,)
