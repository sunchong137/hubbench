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
import sys
sys.path.append("../")
import hubbard_ba, hubbard_fci
Pi = np.pi

def test_arrays():
    L = 10
    Nup = 5
    Ndn = 5

    Iarr, Jarr, karr, Larr = hubbard_ba.gen_grids(L, Nup, Ndn)
    I_ref = np.arange(-4.5, 5.5, 1)
    J_ref = np.arange(-2, 3, 1)
    k_ref = np.array([-2/5, -2/5, -1/5, -1/5, 0, 0, 1/5, 1/5, 2/5, 2/5]) * Pi
    L_ref = np.array([-4/5, -2/5, 0, 2/5, 4/5]) * Pi
    
    assert np.allclose(Iarr, I_ref)
    assert np.allclose(Jarr, J_ref)
    assert np.allclose(karr, k_ref)
    assert np.allclose(Larr, L_ref)

def test_lieb_wu():
    L = 10
    Nup = L//2
    Ndn = L//2
    U = 4
    E_ba = hubbard_ba.lieb_wu(L, Nup, Ndn, U)
    E_fci, _ = hubbard_fci.hubbard_fci(L, U, nelec=(Nup, Ndn), pbc=True, filling=1.0)
    assert np.allclose(E_ba, E_fci)
