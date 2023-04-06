import numpy as np
import axion_cosmo as ac

#They quote 2.27e14 I think in conclusion
ac.evolve_PQ(shape=(10,10,10),
             fa=2.27e14,
             init_field=lambda shape: 1+np.zeros((2,)+shape),
             init_fieldp=lambda shape: np.zeros((2,)+shape),
             etaini=0.0001,
             deta=0.0001,
             Nstep=62500,
             name="point",
             logdir="/Users/joshlin/axion_cosmo/tests/0_PQ/data",
             tolog=[("point",lambda f,fp:np.average(np.abs(f)),100)],
             flush=100,
             debug=True)
