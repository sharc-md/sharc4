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
    D^IJ = (D^I0)^T * D^J0 - D^I0 * (D^J0)^T
    """

    DIJ_elec = np.einsum('ij,ik,kl->jl', tdmI, Sao, tdmJ, optimize=True, casting='no')
    DIJ_hole = np.einsum('ij,ik,lk->jl', tdmJ, Sao, tdmI, optimize=True, casting='no')

    res = DIJ_elec
    res[:DIJ_hole.shape[0], :DIJ_hole.shape[1]] -= DIJ_hole

    return res
