Hubbard model benchmarking
==========================
A collection of Bethe Ansatz, FCI, DMRG methods for the Hubbard model.
- Author: Chong Sun <sunchong137@gmail.com>

Dependencies:
-------------
- Bethe Ansatz only depends on Numpy and Scipy.
- FCI calculation depends on PySCF.
- DMRG depends on PyBlock3 or Block2.
One can install the latest version from source by

`pip install block2==0.5.3rc5 --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/`

Make sure to go to the GitHub to check the latest version!


2D Data
-------
The `SimonsHub2D` directory contains benchmark data for the 2D Hubbard model 
with a variaty of lattice sizes, doping, frustration and temperature.

The data is obtained from LeBlanc et. al., Phys. Rev. X 5 041041
