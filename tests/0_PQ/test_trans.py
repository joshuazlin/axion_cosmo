import numpy as np
import axion_cosmo as ac

#They quote 2.27e14 I think in conclusion
ac.evolve_PQ(shape=(100,100,10),
             fa=2.27e20,
             init_field=lambda shape: 1+np.zeros((2,)+shape),#lambda shape:ac.dummy_thermal(shape,1e0),
             init_fieldp=lambda shape: 1+np.zeros((2,)+shape),#lambda shape:ac.dummy_thermal(shape,1e0),
             etaini=280,
             deta=0.004,
             Nstep=62500,
             name="test",
             logdir="/Users/joshlin/axion_cosmo/tests/0_PQ/data",
             logstep=1000,
             debug=True)
