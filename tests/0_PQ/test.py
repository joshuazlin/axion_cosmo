
import axion_cosmo as ac

ac.evolve_PQ((5,5),2.27e14,ac.dummy_thermal,ac.dummy_thermal,10000,
             name="test",
             logdir="/Users/joshlin/axion_cosmo/tests/0_PQ/data",
             logstep=10)
