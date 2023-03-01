#!/usr/bin/env python3
"""
https://github.com/mdtraj/mdtraj/blob/0dde9a64563faeec742b563e25deea988edf3c70/mdtraj/geometry/src/sasa.cpp
based on c++ in MDtraj code with GNU license
Singh, Kollman J. Comp. Chem. 1984, 5, 129 - 145
Shrake, Rupley J Mol Biol. 79 (2): 351-71
"""

#  Calculate the accessible surface area of each atom in a single snapshot
#
#  Parameters
#  ----------
#  xyz : 2d array, shape=[n_atoms, 3]
#      The coordinates of the nuclei
#  atom_radii : dict, {'He': 1.40}
#      the van der Waals radii of the atoms PLUS the probe radius
#      updates the radii given in constants (Angstrom)
#  atom_radii : 1d array, shape=[n_atoms]
#      the van der Waals radii of the atoms PLUS the probe radius
#
#  grid_mesh : 2d array, shape=[n_points, 3]

import numpy as np
from lebedev_grids import LEBEDEV
from utils import euclidean_distance_einsum
from functools import reduce
from itertools import chain

lebedev = LEBEDEV()
lebedev_grid = lebedev.load


def random_sphere(n_points, radius=1) -> np.ndarray:
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = phi * (1 + 5**0.5) * indices

    x, y, z = radius * np.cos(theta) * np.sin(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(phi)

    return np.asarray([x, y, z]).transpose()


def golden_sphere(n_points) -> np.ndarray:    # golden sprial from stack overflow
    #
    #  Compute the coordinates of points on a sphere using the
    #  Golden Section Spiral algorithm.
    #
    #  Parameters
    #  ----------
    #  n_pts : int
    #      Number of points to generate on the sphere
    inc = np.pi * (3.0 - np.sqrt(5.0))
    offset = 2.0 / n_points
    constant = -1.0 + (offset / 2.0)
    indices = np.arange(0, n_points, dtype=float)
    y = offset * indices + constant
    phi = inc * indices
    r = np.sqrt(1.0 - y * y)
    return np.asarray([np.cos(phi) * r, y, np.sin(phi) * r]).transpose()


def gamess_surface(n):
    """Computes approximately n points on unit sphere. Code adapted from GAMESS.

    Parameters
    ----------
    n : int
        approximate number of requested surface points

    Returns
    -------
    ndarray
        numpy array of xyz coordinates of surface points
    """

    u = []
    eps = 1e-10
    nequat = int(np.sqrt(np.pi * n))
    nvert = int(nequat / 2)
    nu = 0
    for i in range(nvert + 1):
        fi = np.pi * i / nvert
        z = np.cos(fi)
        xy = np.sin(fi)
        nhor = int(nequat * xy + eps)
        nhor -= nhor % 2    # this line was added to only allow even numbered horizontal layers
        if nhor < 1:
            nhor = 1
        for j in range(nhor):
            fj = 2 * np.pi * j / nhor
            x = np.cos(fj) * xy
            y = np.sin(fj) * xy
            if nu >= n:
                return np.array(u)
            nu += 1
            u.append([x, y, z])
    return np.array(u)


def markus_deserno(n):
    # https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    a = 4 * np.pi / n
    d = np.sqrt(a)
    M = int(np.pi / d)
    d_t = np.pi / M
    d_p = a / d_t
    points = []
    for m in range(M):
        t = np.pi * (m + 0.5) / M
        M_p = int(2 * np.pi * np.sin(t) / d_p)
        for m2 in range(M_p):
            p = 2 * np.pi * m2 / M_p
            points.append([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    return np.array(points)


def shrake_rupley(
    xyz: np.ndarray,
    atom_radii: np.ndarray,
    out_points: np.ndarray,
    density=1,
    n_points=0,
    grid=lebedev_grid,
    weights=None
) -> np.ndarray:
    natoms = int(xyz.shape[0])
    n_out_points = n_points
    for i in range(natoms):
        if weights is not None:
            sphere_grid, sphere_weights = grid(int(4.0 * np.pi * atom_radii[i]**2 * density))
        else:
            sphere_grid = grid(int(4.0 * np.pi * atom_radii[i]**2 * density))

        sphere_grid = sphere_grid * atom_radii[i] + xyz[i, :]    # scale an shift center
        dist = euclidean_distance_einsum(xyz, sphere_grid)
        invalid = np.concatenate([np.where(dist[iv, :] < v - 0.01)[0] for iv, v in enumerate(atom_radii)]).reshape((-1))
        invalid = np.sort(np.unique(invalid, axis=0))
        idx = np.ones(dist.shape[1], bool)
        idx[invalid] = 0
        last_new = n_out_points + sphere_grid.shape[0] - len(invalid)
        out_points[n_out_points:last_new, :] = sphere_grid[idx, :]
        if weights is not None:
            weights[n_out_points:last_new] = sphere_weights[idx]
        n_out_points = last_new

    return n_out_points


def mk_layers(
    xyz: np.ndarray, atom_radii: list[float], density=1, shells=[1.4, 1.6, 1.8, 2.0], grid='lebedev'
) -> np.ndarray:
    """
    returns the Merz-Kollmann layers for a molecule
    ------
    Parameters:

    xyz: ndarray  cartesian coordinates of each atom in the molecule
    atom_radii: list[float] list of the van-der-Waals radii of each atom (same order as xyz)
    density: float  surface density of each calculated sphere (density of 1. -> 4*pi*r^2)
    shells: list[float] give the scaling factors for each shell
    grid: str specify a quadrature function from 'lebedev', 'random', 'golden_spiral', 'gamess', 'marcus_deserno'
    """
    # guess the number of points generously (lebedev grid requires more points than density!!
    n_points = 2 * int(reduce(lambda acc, x: acc + 4 * np.pi * density * x**2, chain(*map(lambda x: [x * s for s in shells], atom_radii))))
    atom_radii_array = np.array(atom_radii)
    # allocate the memory for the points
    mk_layers_points = np.ndarray((n_points, 3), dtype=float)
    grid_functions = {
        'lebedev': lebedev_grid,
        'random': random_sphere,
        'golden_spiral': golden_sphere,
        'gamess': gamess_surface,
        'marcus_deserno': markus_deserno
    }
    assert grid in grid_functions

    weights = None
    if grid == 'lebedev':
        weights = np.ndarray((n_points), dtype=float)
    grid = grid_functions[grid]
    # potentially parallelizable! every layer is one process
    n_points = 0
    for y in shells:
        n_points = shrake_rupley(
            xyz, y * atom_radii_array, mk_layers_points, density=density, n_points=n_points, grid=grid, weights=weights
        )
    return mk_layers_points[:n_points, :], weights[:n_points] if weights is not None else None
