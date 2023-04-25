# Cosmology Final Project

Final Project for Harvard Cosmology class, taught by Cora Dvorkin. Reproduction of [this paper](https://arxiv.org/abs/1906.00967). You can read our [final paper here](axion_cosmo.pdf), and read our [presentation slides here](presentation/axion_cosmo.pdf).

## Installation Instructions

In a python environment, run the command 
```
pip install -e .
```
in the home directory of this folder. tests/ includes some test python scripts that you can then run from command line to generate data. 
 - generate_field.py     : From command line, run "python generate_field.py [dimx] [dimy] [dimz]" <br>
                           e.g. python generate_field.py 100 100 100 <br>
                           Outputs two .hdf5 files, one for the PQ part of the evolution and one for the QCD part of the evolution <br>
                           e.g. "PQ_ev_(100, 100, 100)\_0.hdf5" and "QCD_ev_(100, 100, 100)\_0.hdf5" <br>
                           You may investigate these files if you wish, or simply use them to generate the topological and energy features <br>
 - generate_features.py  : From command line, run "python generate_features.py [era ('PQ' or 'QCD')] [evolution file name] [timestamp]" <br>
                           e.g. python generate_features.py QCD "QCD_ev_(100, 100, 100)_0" 20 <br>
                           Outputs one .hdf5 file that includes meaningful features from the given era <br>
                           e.g. "PQ_features.hdf5" <br>
                           Then you can use notebooks/examples.ipynb to make visualizations! <br>

## Modules (contained in the src/ directory)
 - thermal.py       : Generates initial thermal configuration
 - evolve_utils.py  : Utilities for evolving the config 
 - evolve.py        : Wrapper for actually evolving, with physical params. Generates .hdf5 files with field values and physical params as keys
 - measure_util.py  : Utilities for identifying topological features (strings & domain walls) and calculating energy density after QCD phase transition
 - measure.py       : Wrapper for generating .hdf5 files that contain features. Reads files generated by evolve.py
 
 
 
