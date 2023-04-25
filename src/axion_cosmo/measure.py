"""
given complex field value in hdf5 files and time stamp
saves the identified topological structures to file
"""

from .evolve_utils import *
from .params import *
from .thermal import *
from .measure_utils import *
import h5py

def measure(timestamp=-1,
              readdir=None,
              filename=None,
              savedir=None,
              savename=None,
              era=None
              ):
    """
    Inputs
   timestamp : index of time to take from evolution file
     readdir : directory of evolution file
    filename : file to read
     savedir : directory of save string/domainwall/density file
    savename : name for saved files
         era : either 'PQ' or 'QCD'
    """


    f = h5py.File(f'{readdir}{filename}.hdf5','r')
    field = f['field'][timestamp][0]

    f_features = h5py.File(f'{savedir}{savename}.hdf5','w')
    f_features.create_dataset('strings', data=stringid(field))
    f_features.create_dataset('dws', data=domainid(field))

    if era == 'QCD':
        fieldp = f['field'][timestamp][1]
        ma = f['field'].attrs['ma']
        fa = f['field'].attrs['fa']
        scale = f['field'].attrs['set_scale']
        if timestamp == -1:
            eta = f['field'].attrs['eta_QCD'] + (f['field'].attrs['Nstep']*f['field'].attrs['deta'])
        else:
            eta = f['field'].attrs['eta_QCD'] + (timestamp + 1) * f['field'].attrs['deta']
        f_features.create_dataset('density', data=energy_density(field,fieldp,ma,fa,eta,scale))
    

    f_features.close()



    return
