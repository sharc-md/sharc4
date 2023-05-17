#!/usr/bin/env python3
'''
This file contains logic for transforming two TDM between two excited states
and the same reference state into a TDM between the two excited states.
'''

import numpy as np


def es2es_tdm(tdmI, tdmJ, Sao) -> np.ndarray:
    """
    Compute approximate transition density with respect to a reference excited state.
    This is computed analogously to the electron/hole densities:
    D^IJ = (D^0I)^T * D^0J - D^0J * (D^0I)^T
    """

    DIJ_elec = tdmI.T @ Sao @ tdmJ
    DIJ_hole = tdmJ @ Sao @ tdmI.T

    res = DIJ_elec
    res[:DIJ_hole.shape[0], :DIJ_hole.shape[1]] -= DIJ_hole

    return res
