#!/usr/bin/env python3
"""
This file contains logic for transforming two TDM between two excited states
and the same reference state into a TDM between the two excited states.
"""

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2025 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************


import numpy as np


def es2es_tdm(dens_0i, dens_0j, Sao) -> np.ndarray:
    """
    Compute approximate transition density with respect to a reference excited state.
    This is computed analogously to the electron/hole densities:
    $D^IJ = (D^0I)^T * D^0J - D^0J * (D^0I)^T$ <- this is a commutator

    Overlap in between is necessary because $C*C^T = S$
    """

    DIJ_elec = dens_0i.T @ Sao @ dens_0j
    DIJ_hole = dens_0j @ Sao @ dens_0i.T

    return DIJ_elec - DIJ_hole
