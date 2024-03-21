#!/usr/bin/env python3
"""
This file contains logic for transforming two TDM between two excited states
and the same reference state into a TDM between the two excited states.
"""

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
