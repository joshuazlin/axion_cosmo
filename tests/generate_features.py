
import axion_cosmo as ac
import numpy as np
import sys
import h5py



# command line input of era of evolution and evolution file name to read
era = sys.argv[1]
filename = sys.argv[2]
timestamp = sys.argv[3]


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
ac.measure(timestamp=int(timestamp),
           readdir='./',
           filename=filename,
           savedir='',
           savename=f'{era}_{timestamp}_features',
           era=era)

print("Saved files!")




