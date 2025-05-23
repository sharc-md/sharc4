#!/usr/bin/env python3

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

# Short program to evaluate partial charges with the restircted electrostatic potential fit
# Source Phys.Chem.Chem.Phys.,2019,21, 4082--4095 cp/c8cp06567e and TheJournalofPhysicalChemistry,Vol.97,No.40,199 10.1021/j100142a004
#  gaussian can do RESP
from error import Error
from globals import DEBUG

import sys
import numpy as np
from asa_grid import mk_layers
from pyscf import gto, df
from pyscf.lib import misc
from constants import au2a
from logger import log

#  np.set_printoptions(threshold=sys.maxsize, linewidth=10000, precision=1, formatter={"float": lambda x: f"{x: 1.0f}"})


def get_resp_grid(atom_radii: list[int], coords: np.ndarray, density=1, shells=[1.4, 1.6, 1.8, 2.0], grid="lebedev"):
    return mk_layers(coords, atom_radii, density, shells, grid)


class Resp:
    def __init__(
        self,
        coords: np.ndarray,
        atom_symbols: list[str],
        atom_radii: list[float],
        density=1,
        shells=[1.4, 1.6, 1.8, 2.0],
        custom_grid: np.ndarray = None,
        grid="lebedev",
        logger=log,
        generate_raw_fit_file=False,
    ):
        """
        creates an object with a fitting grid and precalculated properties for the molecule.

        Parameters:
        ----------
        coords: ndarray array with shape (natom,3) unit has to be Bohr

        atom_symbols: list[str] with atom symbols order corresponding to coord

        density: int specifying the surface density on the spheres in the Merz-Kollman scheme

        shells: list[int] with factors for each shell in the Merz-Kollman scheme

        custom_grid: ndarray[natoms, 3] defining a grid to fit on to (overwrites grid keyword)

        grid: string specify a quadrature function from 'lebedev', 'random', 'golden_spiral', 'gamess', 'marcus_deserno'

        logger: python.logging.log object to log to default is logger.log

        generate_raw_fit_file: bool  generate file with ESP data (unit: au) [RESP_fit_data.txt]! [num ref_esp fit_esp dist x y z]
        """
        self.log = logger
        self.coords = coords
        self.atom_symbols = atom_symbols
        self.mk_grid = custom_grid
        self.weights = None
        self.generate_raw_fit_file = generate_raw_fit_file
        if self.mk_grid is None:
            self.mk_grid, self.weights = get_resp_grid(atom_radii, coords * au2a, density, shells, grid)
            self.mk_grid /= au2a  # convert back to Bohr
        assert len(self.mk_grid.shape) == 2 and self.mk_grid.shape[1] == 3
        self.natom = coords.shape[0]
        self.ngp = self.mk_grid.shape[0]
        self.log.info(f"Done initializing grid.\n\tgrid points per atom:{self.ngp/self.natom}")
        # Build 1/|R_A - r_i| m_A_i
        self.R_alpha: np.ndarray = np.full((self.natom, self.ngp, 3), self.mk_grid) - self.coords[:, None, :]  # rA-ri
        self.r_inv: np.ndarray = 1 / np.sqrt(np.sum((self.R_alpha) ** 2, axis=2))  # 1 / |ri-rA|

    def prepare(self, mol: gto.Mole, ncpu=1):
        """
        Prepares the gto.Mole object and 3c2e integrals

        Sets up the gto.Mole object for calculations and also precalcualtes all 3d2e integrals for the evaluation of ESPs.

        Parameters
        ----------
        mol: gto.Mole
        ncpu: int
        """
        Z = mol.atom_charges()
        self.log.trace(f"{self.natom} {Z} {self.r_inv.shape}")
        self.Sao = mol.intor("int1e_ovlp")
        self.Vnuc = np.sum(Z[..., None] * self.r_inv, axis=0)
        fakemol = gto.fakemol_for_charges(self.mk_grid)
        # NOTE This could be very big (fakemol could be broken up into multiple pieces)
        # NOTE the value of these integrals is not affected by the atom charge
        self.log.info("starting to evaluate integrals")

        with misc.with_omp_threads(ncpu) as _:
            self.ints = df.incore.aux_e2(mol, fakemol, intor="int3c2e")
        self.log.info("done")

    def one_shot_fit(self, dm: np.ndarray, include_core_charges: bool, order=2, charge=0, **kwargs):
        """
        fits RESP charges onto a density matrix in AO basis up to given order (all at once)

        calculates the electrostatic potential of the density at the grid points with 2e3c integrals using pyscf.
        uses this ESP as a reference for the RESP fitting procedure.

        Parameters
        ----------
        dm : np.ndarray
            density matrix in AO basis
        include_core_charges : bool
            whether to include nuclear charges of the atoms (turn off for transition densities)
        order : int
            fitting order (0: monopoles, 2: dipoles, 3: quadrupoles)
        """
        self.log.debug("one shot fit start")
        if not (0 <= order <= 2):
            raise Error("Specify order in the range of 0 - 2")
        n_fits = sum([1, 3, 6][: order + 1])
        natom = self.natom
        Vnuc = np.copy(self.Vnuc) if include_core_charges else np.zeros((self.ngp), dtype=float)
        Vele = np.einsum("ijp,ij->p", self.ints, dm)
        Fesp_i = Vnuc - Vele
        R_alpha = self.R_alpha
        r_inv = self.r_inv
        self.r_inv3 = self.rinv3 if "rinv3" in self.__dict__ else r_inv**3
        self.r_inv5 = self.rinv5_2 if "rinv5_2" in self.__dict__ else r_inv**5

        if order == 2:
            tmp = np.vstack(
                (
                    self.r_inv,  # .
                    R_alpha[:, :, 0] * self.r_inv3,  # x
                    R_alpha[:, :, 1] * self.r_inv3,  # y
                    R_alpha[:, :, 2] * self.r_inv3,  # z
                    R_alpha[:, :, 0] * R_alpha[:, :, 0] * self.r_inv5 * 0.5,  # xx
                    R_alpha[:, :, 1] * R_alpha[:, :, 1] * self.r_inv5 * 0.5,  # yy
                    R_alpha[:, :, 2] * R_alpha[:, :, 2] * self.r_inv5 * 0.5,  # zz
                    R_alpha[:, :, 0] * R_alpha[:, :, 1] * self.r_inv5,  # xy
                    R_alpha[:, :, 0] * R_alpha[:, :, 2] * self.r_inv5,  # xz
                    R_alpha[:, :, 1] * R_alpha[:, :, 2] * self.r_inv5,  # yz
                )
            )  # m_A_i
        elif order == 1:
            tmp = np.vstack(
                (
                    self.r_inv,  # .
                    R_alpha[:, :, 0] * self.r_inv3,  # x
                    R_alpha[:, :, 1] * self.r_inv3,  # y
                    R_alpha[:, :, 2] * self.r_inv3,  # z
                )
            )  # m_A_i
        elif order == 0:
            tmp = self.r_inv  # m_A_i
        n_af = natom * n_fits
        dim = n_af + 1

        if self.weights is not None:
            a = np.einsum("ag,g,bg->ab", tmp, self.weights, tmp)
            b = np.einsum("ag,g,g->a", tmp, self.weights, Fesp_i)  # v_A
        else:
            a = np.einsum("ag,bg->ab", tmp, tmp)
            b = np.einsum("ag,g->a", tmp, Fesp_i)  # v_A

        A = np.zeros((dim, dim))
        A[:-1, :-1] += a
        A[:natom, -1] = 1.0
        A[-1, :natom] = 1.0

        B = np.zeros((dim))
        B[:-1] += b
        B[-1] = float(charge)

        beta_au = self.beta  # needs to be 1/au**2
        #  beta_au = 0.05    # needs to be 1/au**2

        def get_rest(Q, b=0.1):
            return -beta_au / (np.sqrt(Q**2 + b**2))

        vget_rest = np.vectorize(get_rest, cache=True)

        # solve ESP monopoles
        A_mon = np.copy(A[: natom + 1, : natom + 1])
        A_mon[-1, :natom] = 1.0
        B_mon = np.copy(B[: natom + 1])
        B_mon[-1] = float(charge)
        Q_last = np.linalg.solve(A_mon, B_mon)[:natom]
        self.log.debug(f"ESP Monopoles:\n\t{Q_last}")
        Q_new = np.ones(Q_last.shape, dtype=float)

        max_iterations = 500
        iteration = 0
        while np.linalg.norm(Q_last - Q_new) >= 0.00001 and iteration < max_iterations:
            #  rest = vget_rest(Q_last) * au2a**2
            rest = vget_rest(Q_last, b=0.1)
            Q_last = Q_new.copy()
            A_rest = np.copy(A_mon)
            B_rest = np.copy(B_mon)
            # add restraint to B
            #  B_rest[:n_af] += target * rest
            np.einsum("ii->i", A_rest)[:natom] -= rest
            Q_new = np.linalg.solve(A_rest, B_rest)[:natom]
            iteration += 1

        target = np.zeros(n_af, dtype=float)
        # potentially alter targets
        # restrain to RESP monopoles
        target[:natom] = Q_new
        self.log.debug(f"TARGET CHARGES:\n\t{target.reshape((n_fits, -1)).T}")

        # set initial guess to resp monopoles
        Q_last = np.zeros((n_af), dtype=float)
        Q_last[:natom] = Q_new
        #  Q_last = np.linalg.solve(A, B)[:n_af]
        Q_new = np.ones(Q_last.shape, float)

        rest = np.zeros((B.shape))
        max_iterations = 500
        iteration = 0
        while np.linalg.norm(Q_last - Q_new) >= 0.00001 and iteration < max_iterations:
            #  rest = vget_rest(Q_last) * au2a**2
            rest = vget_rest(Q_last, b=0.1)
            Q_last = Q_new.copy()
            A_rest = np.copy(A)
            B_rest = np.copy(B)
            # add restraint to B
            np.einsum("ii->i", A_rest)[:n_af] -= rest
            B_rest[:n_af] += target * rest
            Q_new = np.linalg.solve(A_rest, B_rest)[:n_af]
            iteration += 1
        self.log.print("exciting RESP fitting loop after", iteration, " iterations. Norm", np.linalg.norm(Q_last - Q_new))

        fit_esp = np.einsum("x,xi->i", Q_new, tmp)
        residual_ESP = fit_esp - Fesp_i
        res = Q_new.reshape((n_fits, -1)).T

        self.log.info(
            f"Fit done!\tMEAN: {np.mean(residual_ESP): 10.6e}\t ABS.MEAN: {np.mean(np.abs(residual_ESP)): 10.6e}\tRMSD: {np.sqrt(np.mean(residual_ESP**2)): 10.8e}"
        )
        if self.generate_raw_fit_file:
            filename = kwargs["resp_data_file"] if "resp_data_file" in kwargs else "RESP_fit_data.txt"
            dist = np.min(np.linalg.norm(R_alpha, axis=2), axis=0)
            self.log.info("generating file with ESP data [filename]! [num ref_esp fit_esp dist x y z]")
            with open(filename, "w") as f:
                f.write(f"# {'num [AU]':<9s}  {'ref_esp':<12s} {'fit_esp':<12s} {'dist':<12s} {'x':<12s} {'y':<12s} {'z':<12s}\n")
                for i, vals in enumerate(zip(Fesp_i, fit_esp, dist, self.mk_grid[:, 0], self.mk_grid[:, 1], self.mk_grid[:, 2])):
                    f.write(f"{i:5d}      " + " ".join(map(lambda x: f"{x: 12.8f}", vals)) + "\n")

        # make traceless (Source: Sebastian)
        traces = np.sum(res[:, 4:7], axis=1)
        res[:, 4:7] -= 1 / 3 * traces[..., None]
        self.log.debug("one shot fit finish")

        return res

    def sequential_multipoles(
        self, dm: np.ndarray, include_core_charges=True, charge=0, order=2, betas=[0.0005, 0.0015, 0.003], **kwargs
    ):
        self.log.info("Start sequential multipolar fit")
        if not include_core_charges and charge != 0:
            self.log.warning(
                "No core charges but charge not set to zero! -> transition densities set 'include_core_charges=False' and 'charge=0'"
            )
        if not (0 <= order <= 2):
            raise Error("Specify order in the range of 0 - 2")
        natom = self.natom
        Vnuc = np.copy(self.Vnuc) if include_core_charges else np.zeros((self.ngp), dtype=float)
        Vele = np.einsum("ijp,ij->p", self.ints, dm)
        Fesp_i = Vnuc - Vele
        R_alpha = self.R_alpha
        r_inv = self.r_inv

        # fit monopoles
        monopoles, Fres = self.fit(r_inv, Fesp_i, 1, natom, beta=betas[0], charge=charge, weights=self.weights)

        if order == 0:
            return monopoles

        if not hasattr(self, "r_inv3"):
            self.r_inv3 = r_inv**3

        # fit dipoles
        tmp = np.vstack((R_alpha[:, :, 0] * self.r_inv3, R_alpha[:, :, 1] * self.r_inv3, R_alpha[:, :, 2] * self.r_inv3))  # m_A_i
        dipoles, Fres = self.fit(tmp, Fres, 3, natom, beta=betas[1], weights=self.weights, charge=None)

        if order == 1:
            return np.hstack((monopoles, dipoles))

        if not hasattr(self, "r_inv5"):
            self.r_inv5 = r_inv**5

        tmp = np.vstack(
            (
                R_alpha[:, :, 0] * R_alpha[:, :, 0] * self.r_inv5 * 0.5,
                R_alpha[:, :, 1] * R_alpha[:, :, 1] * self.r_inv5 * 0.5,
                R_alpha[:, :, 2] * R_alpha[:, :, 2] * self.r_inv5 * 0.5,
                R_alpha[:, :, 0] * R_alpha[:, :, 1] * self.r_inv5,
                R_alpha[:, :, 0] * R_alpha[:, :, 2] * self.r_inv5,
                R_alpha[:, :, 1] * R_alpha[:, :, 2] * self.r_inv5,
            )
        )  # m_A_i

        quadrupoles, Fres = self.fit(tmp, Fres, 6, natom, beta=betas[2], charge=None, weights=self.weights, traceless_quad=True)

        self.log.info(
            f"Fit done!\tMEAN: {np.mean(Fres): 10.6e}\t ABS.MEAN: {np.mean(np.abs(Fres)): 10.6e}\tRMSD: {np.sqrt(np.mean(Fres**2)): 10.8e}"
        )
        if self.generate_raw_fit_file:
            filename = kwargs["resp_data_file"] if "resp_data_file" in kwargs else "RESP_fit_data.txt"
            dist = np.min(np.linalg.norm(R_alpha, axis=2), axis=0)
            self.log.info(f"generating file with ESP data [{filename}]! [num ref_esp fit_esp dist x y z]")
            with open(filename, "w") as f:
                f.write(f"# {'num [AU]':<9s}  {'ref_esp':<12s} {'fit_esp':<12s} {'dist':<12s} {'x':<12s} {'y':<12s} {'z':<12s}\n")
                for i, vals in enumerate(
                    zip(Fesp_i, Fesp_i - Fres, dist, self.mk_grid[:, 0], self.mk_grid[:, 1], self.mk_grid[:, 2])
                ):
                    f.write(f"{i:5d}      " + " ".join(map(lambda x: f"{x: 12.8f}", vals)) + "\n")

        return np.hstack((monopoles, dipoles, quadrupoles))

    @staticmethod
    def fit(tmp, Fesp_i, n_fits, natom, charge=None, beta=0.0005, b_par=0.1, weights=None, traceless_quad=False, logger=None):
        return _fit(
            tmp,
            Fesp_i,
            n_fits,
            natom,
            charge=charge,
            beta=beta,
            b_par=b_par,
            weights=weights,
            traceless_quad=traceless_quad,
            logger=logger,
        )

    def prepare_parallel(self, densities_dict, order=2, **kwargs):
        global fit_data
        fit_data = {}
        # self.log.info("Start sequential multipolar fit")
        if not (0 <= order <= 2):
            raise Error("Specify order in the range of 0 - 2")
        fit_data["densities_dict"] = densities_dict
        R_alpha = self.R_alpha
        fit_data["R_alpha"] = self.R_alpha
        fit_data["ints"] = self.ints
        fit_data["Vnuc"] = self.Vnuc
        fit_data["weights"] = self.weights
        fit_data["geo_tens0"] = self.r_inv

        if order == 0:
            return

        r_inv3 = self.r_inv**3

        # fit dipoles
        fit_data["geo_tens1"] = np.vstack(
            (R_alpha[:, :, 0] * r_inv3, R_alpha[:, :, 1] * r_inv3, R_alpha[:, :, 2] * r_inv3)
        )  # m_A_i

        if order == 1:
            return

        r_inv5 = self.r_inv**5

        fit_data["geo_tens2"] = np.vstack(
            (
                R_alpha[:, :, 0] * R_alpha[:, :, 0] * r_inv5 * 0.5,
                R_alpha[:, :, 1] * R_alpha[:, :, 1] * r_inv5 * 0.5,
                R_alpha[:, :, 2] * R_alpha[:, :, 2] * r_inv5 * 0.5,
                R_alpha[:, :, 0] * R_alpha[:, :, 1] * r_inv5,
                R_alpha[:, :, 0] * R_alpha[:, :, 2] * r_inv5,
                R_alpha[:, :, 1] * R_alpha[:, :, 2] * r_inv5,
            )
        )  # m_A_i

        return

    multipoles_from_dens = sequential_multipoles
    #  multipoles_from_dens = one_shot_fit


def multipoles_from_dens_parallel(
    dm_key: tuple, include_core_charges=True, charge=0, order=2, betas=[0.0005, 0.0015, 0.003], natom=None, target=None
):
    # self.log.info("Start sequential multipolar fit")
    if not include_core_charges and charge != 0:
        raise RuntimeError()

    if not (0 <= order <= 2):
        raise RuntimeError("Specify order in the range of 0 - 2")

    dm = fit_data["densities_dict"][dm_key]
    Fesp_i = -np.einsum("ijp,ij->p", fit_data["ints"], dm)

    if include_core_charges:
        Fesp_i += fit_data["Vnuc"]

    # fit monopoles
    monopoles, Fres = _fit(
        fit_data["geo_tens0"], Fesp_i, 1, natom, beta=betas[0], charge=charge, weights=fit_data["weights"], target=target
    )

    if order == 0:
        return monopoles

    dipoles, Fres = _fit(fit_data["geo_tens1"], Fres, 3, natom, beta=betas[1], weights=fit_data["weights"], charge=None)

    if order == 1:
        return np.hstack((monopoles, dipoles))

    quadrupoles, Fres = _fit(
        fit_data["geo_tens2"], Fres, 6, natom, beta=betas[2], charge=None, weights=fit_data["weights"], traceless_quad=True
    )

    # self.res[(dm_key[0], dm_key[1])] = np.hstack((monopoles, dipoles, quadrupoles))
    return np.hstack((monopoles, dipoles, quadrupoles))


def _fit(
    tmp, Fesp_i, n_fits, natom, charge=None, beta=0.0005, b_par=0.1, weights=None, traceless_quad=False, logger=None, target=None
):
    n_af = natom * n_fits
    dim = n_af + 1

    if weights is not None:
        a = np.einsum("ag,g,bg->ab", tmp, weights, tmp)
        b = np.einsum("ag,g,g->a", tmp, weights, Fesp_i)  # v_A
    else:
        a = np.einsum("ag,bg->ab", tmp, tmp)
        b = np.einsum("ag,g->a", tmp, Fesp_i)  # v_A

    if charge is not None:
        A = np.zeros((dim, dim))
        A[:-1, :-1] += a
        A[:natom, -1] = 1.0
        A[-1, :natom] = 1.0

        B = np.zeros((dim))
        B[:-1] += b
        B[-1] = float(charge)
    elif traceless_quad:
        # traceless_quadrupoles constraint for every atom
        dim = n_af + natom
        A = np.zeros((dim, dim))
        A[:n_af, :n_af] = a
        B = np.zeros((dim))
        B[:n_af] = b

        for ia in range(natom):
            idx = [ia + j * natom for j in range(3)]
            A[idx, n_af + ia] = 1.0
            A[n_af + ia, idx] = 1.0
            B[n_af + ia] = 0.0
    else:
        A = a
        B = b

    def get_rest(Q):
        return -beta / (np.sqrt(Q**2 + b_par**2))

    vget_rest = np.vectorize(get_rest, cache=True)

    Q_last = np.linalg.solve(A, B)[:n_af]
    Q_new = np.ones(Q_last.shape)
    max_iterations = 500
    iteration = 0
    while np.linalg.norm(Q_last - Q_new) >= 0.00001 and iteration < max_iterations:
        if target is not None:
            Q_last = Q_last - target
        rest = vget_rest(Q_last)
        Q_last = Q_new.copy()
        A_rest = np.copy(A)
        # add restraint to B
        np.einsum("ii->i", A_rest)[:n_af] -= rest
        Q_new = np.linalg.solve(A_rest, B)[:n_af]
        iteration += 1

    n2order = {1: "monopoles", 3: "dipoles", 6: "quadrupoles"}
    if logger is not None and type(logger) is callable:
        logger(f"exciting RESP fit for {n2order[n_fits]} after {iteration} iterations. \tNorm: {np.linalg.norm(Q_last - Q_new)}")

    fit_esp = np.einsum("x,xi->i", Q_new, tmp)
    residual_ESP = fit_esp - Fesp_i
    if logger is not None and type(logger) is callable:
        logger(
            f"Fit done!, MEAN: {np.mean(residual_ESP): 10.6e}, ABS.MEAN: {np.mean(np.abs(residual_ESP)): 10.6e}, RMSD: {np.sqrt(np.mean(residual_ESP**2))}"
        )
    res = Q_new.reshape((n_fits, -1)).T
    return res, residual_ESP
