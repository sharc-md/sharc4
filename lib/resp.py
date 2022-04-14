#!/usr/bin/env python3

# Short program to evaluate partial charges with the restircted electrostatic potential fit
# Source Phys.Chem.Chem.Phys.,2019,21, 4082--4095 cp/c8cp06567e and TheJournalofPhysicalChemistry,Vol.97,No.40,199 10.1021/j100142a004
#  gaussian can do RESP
from error import Error
from constants import ATOMIC_RADII

import sys
import numpy as np
from asa_grid import mk_layers
from pyscf import gto, df

au2a = 0.52917721092
np.set_printoptions(threshold=sys.maxsize, linewidth=10000, precision=5)


def get_resp_grid(atom_symbols: list[str], coords: np.ndarray, density=1, shells=[1.4, 1.6, 1.8, 2.0]):
    atom_radii = np.fromiter(map(lambda x: ATOMIC_RADII[x], atom_symbols), dtype=float)
    return mk_layers(coords, atom_radii, density, shells)

class Resp:
    def __init__(self, coords: np.ndarray, atom_symbols: list[str], density=1, shells=[1.4, 1.6, 1.8, 2.0], custom_grid: np.ndarray=None):
        """
        creates an object with a fitting grid and precalculated properties for the molecule.

        Parameters:
        ----------
        coords: ndarray array with shape (natom,3) unit has to be Bohr

        atom_symbols: list[str] with atom symbols order corresponding to coord

        density: int specifying the surface density on the spheres in the Merz-Kollman scheme

        shells: list[int] with factors for each shell in the Merz-Kollman scheme

        custom_grid: ndarray[natoms, 3] defining a grid to fit on to
        """
        self.beta = 0.0005
        self.coords = coords
        self.atom_symbols = atom_symbols
        self.mk_grid = custom_grid
        if self.mk_grid is None:
            self.mk_grid = get_resp_grid(atom_symbols, coords * au2a, density, shells) / au2a
        assert len(self.mk_grid.shape) == 2 and self.mk_grid.shape[1] == 3
        self.natom = coords.shape[0]
        self.ngp = self.mk_grid.shape[0]
        # Build 1/|R_A - r_i| m_A_i
        self.R_alpha: np.ndarray = np.full((self.natom, self.ngp, 3), self.mk_grid) - self.coords[:, None, :]  # rA-ri
        self.r_inv: np.ndarray = 1 / np.sqrt(np.sum((self.R_alpha)**2, axis=2))    # 1 / |ri-rA|

    def prepare(self, basis, cart_basis=False):
        natom = len(self.atom_symbols)
        atoms = [[f'{s.upper()}{j+1}', c.tolist()] for j, s, c in zip(range(natom), self.atom_symbols, self.coords)]
        mol = gto.Mole(atom=atoms, basis=basis, unit='BOHR', symmetry=False, cart=cart_basis)
        mol.build()
        Z = mol.atom_charges()
        self.Vnuc = np.sum(Z[..., None] * self.r_inv, axis=0)
        fakemol = gto.fakemol_for_charges(self.mk_grid)
        # NOTE This could be very big (fakemol could be broken up into multiple pieces)
        # NOTE the value of these integrals is not affected by the atom charge
        self.ints = df.incore.aux_e2(mol, fakemol)

    def multipoles_from_dens(self, dm: np.ndarray, include_core_charges: bool, order=2):
        if not (0 <= order <= 2):
            raise Error("Specify order in the range of 0 - 2")
        n_fits = sum([1, 3, 6][:order + 1])
        Fesp_i = np.copy(self.Vnuc) if include_core_charges else np.zeros((self.ngp), dtype=float)
        Vele = np.einsum('ijp,ij->p', self.ints, dm)
        Fesp_i[...] -= Vele
        fits = np.empty((self.natom, n_fits))
        R_alpha = self.R_alpha
        r_inv = self.r_inv
        mp = self.fit_monopoles(Fesp_i)
        if order == 0:
            return mp.reshape((self.natom, 1))
        Fesp_i_res = Fesp_i - mp @ r_inv
        self.r_inv3 = self.rinv3 if 'rinv3' in self.__dict__ else r_inv**3
        dp = self.fit_dipoles(Fesp_i_res).reshape((3, -1))
        if order == 1:
            return np.vstack((mp, dp)).T
        P = np.einsum('xi,inx->in', dp, R_alpha)
        Fesp_i_res = Fesp_i_res - np.einsum('in,in->n', P, self.r_inv3)
        qp = self.fit_quadrupoles(Fesp_i_res).reshape((6, -1))
        fits = np.vstack((mp, dp, qp)).T
        return fits

    @staticmethod
    def _fit(A, B, beta=0.0005, b=0.1, restraint=True):
        Q1 = np.linalg.solve(A, B)
        if not restraint:
            return Q1
        Q2 = np.ones(Q1.shape, float)

        def get_rest(Q):
            return beta / (np.sqrt(Q**2 + b**2))

        vget_rest = np.vectorize(get_rest, cache=True)
        while np.linalg.norm(Q1 - Q2) >= 0.00001:
            Q1 = Q2.copy()
            rest = vget_rest(Q1)
            B_rest = B
            A_rest = A + np.diag(rest)
            Q2 = np.linalg.solve(A_rest, B_rest)
        return Q2

    def fit_monopoles(self, Fesp_i):
        natom = self.natom
        # build B
        b = self.r_inv @ Fesp_i    # v_A
        # build A
        a = self.r_inv @ self.r_inv.T    # contract over ngp -> m_A_A
        # q_A
        A = np.zeros((natom + 1, natom + 1))
        A[:natom, :natom] += a
        A[:natom, natom] = 1
        A[natom, :natom] = 1

        B = np.zeros((natom + 1))
        B[:natom] += b
        B[natom] = 0    # TODO reintroduce charge!!

        charges = self._fit(A, B, self.beta, 0.1)
        return charges[:natom]

    def fit_dipoles(self, Fesp_i: np.ndarray):
        # Build 1/|R_A - r_i| m_A_i
        R_alpha = self.R_alpha

        r_inv3 = self.r_inv3

        tmp = np.vstack((R_alpha[:, :, 0] * r_inv3, R_alpha[:, :, 1] * r_inv3, R_alpha[:, :, 2] * r_inv3))    # m_A_i
        # build A
        A = tmp @ tmp.T    # contract over ngp -> m_A_A

        # build B'
        B = tmp @ Fesp_i    # v_A

        return self._fit(A, B, self.beta, 0.1)

    def fit_quadrupoles(self, Fesp_i):
        natom = self.natom
        # Build 1/|R_A - r_i| m_A_i
        R_alpha = self.R_alpha

        r_inv = self.r_inv
        r_inv5 = self.rinv5 if 'rinv5' in self.__dict__ else r_inv**5
        r_inv5_2 = 0.5 * r_inv5
        # order xx, yy, zz, xy, xz, yz
        tmp = np.vstack(
            (
                R_alpha[:, :, 0] * R_alpha[:, :, 0] * r_inv5_2, R_alpha[:, :, 1] * R_alpha[:, :, 1] * r_inv5_2,
                R_alpha[:, :, 2] * R_alpha[:, :, 2] * r_inv5_2, R_alpha[:, :, 0] * R_alpha[:, :, 1] * r_inv5_2,
                R_alpha[:, :, 0] * R_alpha[:, :, 2] * r_inv5_2, R_alpha[:, :, 1] * R_alpha[:, :, 2] * r_inv5_2
            )
        )    # m_A_i
        # build A
        A = tmp @ tmp.T    # contract over ngp -> m_A_A

        # build B'
        B = tmp @ Fesp_i    # v_A

        quadrupoles = self._fit(A, B, self.beta, 0.1)
        # make traceless (Source: Sebastian)
        quad_mat = quadrupoles.reshape((-1, natom))
        traces = np.sum(quad_mat[:3, :], axis=0)
        quad_mat[:3, :] -= 1 / 3 * traces[None, ...]
        return quad_mat.flatten()
