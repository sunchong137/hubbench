# Copyright 2023 HubBench developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from pyscf import fci
import logging


def hubham_1d(nsite, U, pbc=True, noisy=False, max_w=1.0, spin=0):
    '''
    Return 1D Hubbard model Hamiltonians (h1e and h2e).
    Unit: hopping amplitude t.
    Args:
        nsite : int, number of sites.
        U: double, Hubbard U/t value.
        pbc: boolean, if True, the system obeys periodic boundary condition.
    Returns:
        2D array: h1e
        4D array: h2e
    '''
    h1e = np.zeros((nsite, nsite))
    h2e = np.zeros((nsite,)*4)

    for i in range(nsite-1):
        h1e[i, (i+1)] = -1.
        h1e[(i+1), i] = -1.
        h2e[i,i,i,i] = U
    h2e[-1,-1,-1,-1] = U

    if pbc:
        # assert nsite%4 == 2, "PBC requires nsite = 4n+2!"
        h1e[0, -1] = -1.
        h1e[-1, 0] = -1.

    if noisy:
        if spin == 0:
            noise = (np.random.rand(nsite)*2-1) * max_w
            noise = np.diag(noise)
            h1e += noise 
            return h1e, h2e
        elif spin == 1:
            noise1 = (np.random.rand(nsite)*2-1) * max_w
            noise2 = (np.random.rand(nsite)*2-1) * max_w
            return [h1e+np.diag(noise1), h1e+np.diag(noise2)], h2e
        else:
            raise ValueError("spin has to be 0 or 1.")
    else:
        return h1e, h2e


def hubbard_fci(nsite, U, nelec=None, pbc=True, filling=1.0):
    
    h1e, eri = hubham_1d(nsite, U, pbc)

    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
        if abs(nelec - nsite * filling) > 1e-2:
            logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer number of electrons!".format(filling, nelec/nsite))
    
    # initial guess
    try:
        na, nb = nelec 
    except:
        na = nelec//2
        nb = nelec - na

    if na == nb:
        cisolver = fci.direct_spin0
    else:
        cisolver = fci.direct_spin1
        nelec = (na, nb)
    # len_a = int(special.comb(nsite, na) + 1e-10)
    # len_b = int(special.comb(nsite, nb) + 1e-10)
    # ci0 = np.random.rand(len_a, len_b)
    # ci0 = np.ones((len_a, len_b)) / np.sqrt(len_a*len_b)

    e, c = cisolver.kernel(h1e, eri, nsite, nelec)
    return e, c