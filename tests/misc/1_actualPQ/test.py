import numpy as np
import axion_cosmo as ac

#They quote 2.27e14 I think in conclusion
ac.evolve_PQ(shape=(10,10,10),
             fa=2.27e20,
             init_field=lambda shape: 1e1*np.random.random((2,)+shape),
             init_fieldp=lambda shape: 1e1*np.random.random((2,)+shape),
             etaini=1000,
             deta=0.00001,
             Nstep=10000000,
             name="PQ",
             logdir="/Users/joshlin/axion_cosmo/tests/1_actualPQ/data",
             tolog=[("av",lambda f_fp:np.average(f_fp,axis=(1,2,3)),100,(4,),np.float64),
                    ("absav", lambda f_fp:np.average(np.abs(f_fp),axis=(1,2,3)),100,(4,),np.float64),
                    ("field",lambda f_fp:f_fp,10000,(4,10,10,10),np.float64)],
             flush=100000,
             debug=True)
