Hubbard model benchmarking
==========================
A collection of Bethe Ansatz, FCI, DMRG methods for the Hubbard model.
- Author: Chong Sun <sunchong137@gmail.com>

Dependencies:
-------------
- Bethe Ansatz only depends on Numpy and Scipy.
- FCI calculation depends on PySCF.
- DMRG depends on PyBlock3 or Block2.

All above packages can be installed with `pip`.

2D Data
-------
The `SimonsHub2D` directory contains benchmark data for the 2D Hubbard model 
with a variaty of lattice sizes, doping, frustration and temperature.

The data is obtained from LeBlanc et. al., Phys. Rev. X 5 041041
