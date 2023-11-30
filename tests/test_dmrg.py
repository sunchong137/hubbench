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
import hubbard_pyblock3, hubbard_fci, hubbard_block2

def test_hubbard1d():
    nsite = 6
    U = 4
    E3, mps, mpo = hubbard_pyblock3.hubbard1d_dmrg(nsite, U, pbc=False)
    E_fci, _ = hubbard_fci.hubbard_fci(nsite, U, nelec=nsite, pbc=False)
    E2, _, _ = hubbard_block2.hubbard1d_dmrg(nsite, U, pbc=False)
    assert abs(E3 - E_fci) < 1e-6
    assert abs(E2 - E_fci) < 1e-6
