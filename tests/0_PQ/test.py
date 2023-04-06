import numpy as np
import axion_cosmo as ac

#They quote 2.27e14 I think in conclusion
ac.evolve_PQ(shape=(100,100,10),
             fa=2.27e14,
             init_field=lambda shape: 1e5+1e0*np.random.random((2,)+shape),#ac.dummy_thermal(shape,1e5),
             init_fieldp=lambda shape: np.zeros((2,)+shape),#ac.dummy_thermal(shape,1e1),
             etaini=0.0001,
             deta=0.004,
             Nstep=62500,
             name="test",
             logdir="/Users/joshlin/axion_cosmo/tests/0_PQ/data",
             logstep=1000,
             debug=True)
