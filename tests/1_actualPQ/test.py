import numpy as np
import axion_cosmo as ac

#They quote 2.27e14 I think in conclusion
ac.evolve_PQ(shape=(100,100,10),
             fa=2.27e14,
             init_field=lambda shape: 1+np.zeros((2,)+shape),
             init_fieldp=lambda shape: 1e-5*np.random.random((2,)+shape),
             etaini=0.0001,
             deta=0.0001,
             Nstep=10000000,
             name="PQ",
             logdir="/Users/joshlin/axion_cosmo/tests/1_actualPQ/data",
             tolog=[("av",lambda f_fp:np.average(f_fp,axis=(1,2,3)),100,(4,),np.float64),
                    ("absav", lambda f_fp:np.average(np.abs(f_fp),axis=(1,2,3)),100,(4,),np.float64),
                    ("field",lambda f_fp:f_fp,10000,(4,100,100,10),np.float64)],
             flush=10000,
             debug=False)
