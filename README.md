# Cosmology Final Project

Final Project for Harvard Cosmology class, taught by Cora Dvorkin. Reproduction of [this paper](https://arxiv.org/abs/1906.00967).

Additional References
 - https://arxiv.org/pdf/1202.5851.pdf

Installation Instructions

Run 
  pip install -e .
in the home directory of this folder. tests/ includes some test python scripts that you can then run to see the behaviour. 

Modules (contained in the src/ directory)
 - thermal.py       : Generates initial thermal configuration
 - evolve_utils.py  : Utilities for evolving the config 
 - evolve.py        : Wrappers for actually evolving, with physical params blah blah
 - measure.py       : Measurements of fun physics observables!
 - video.py         : if we want to make it a video

