#!/usr/bin/env python3

import json
import os
import numpy as np
from resp import Resp
from pyscf import gto

# Tests for resp class
INPUTS = os.path.join(os.path.expandvars("$SHARC"), "../tests/interface/inputs")
with open(os.path.join(INPUTS, "resp_input.json")) as f:
    data = json.load(f)
atoms = [[f"{s.upper()}{j+1}", c] for j, s, c in zip(range(len(data["elements"])), data["elements"], data["coords"])]
mol = gto.Mole(
    atom=atoms,
    basis=data["basis"],
    unit="AU",
    charge=0,
    spin=0,
    symmetry=False,
    cart=data["cart_basis"],
    ecp={f'{data["elements"][n]}{n+1}': ecp_string for n, ecp_string in data["ecp"].items()},
)
mol.build()


def test_fit_ch2s():
    fits = Resp(
        np.array(data["coords"]),
        data["elements"],
        data["resp_vdw_radii"],
        data["resp_density"],
        data["resp_shells"],
        grid=data["resp_grid"],
    )
    fits.prepare(mol)
    # first density is 1,1,0, 1,1,0; second 1,1,0, 1,2,0
    for icc, dens, fit in zip([True, False], data["densities"], data["fits"]):
        dens_arr = np.array(dens)
        ref_fit_arr = np.array(fit)
        fit = fits.multipoles_from_dens(
            dens_arr, include_core_charges=icc, order=data["order"], charge=data["charge"], betas=data["betas"]
        )
        np.allclose(ref_fit_arr, fit, rtol=1e-8)
