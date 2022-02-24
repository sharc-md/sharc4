#!/usr/bin/env python3
""" 
https://github.com/mdtraj/mdtraj/blob/0dde9a64563faeec742b563e25deea988edf3c70/mdtraj/geometry/src/sasa.cpp
based on c++ in MDtraj code with GNU license
Singh, Kollman J. Comp. Chem. 1984, 5, 129 - 145
Shrake, Rupley J Mol Biol. 79 (2): 351–71
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


def sphere_grid2(radius, n_points) -> np.ndarray:
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = phi * (1 + 5**0.5) * indices

    x, y, z = radius * np.cos(theta) * np.sin(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(phi)

    return np.asarray([x, y, z]).transpose()


def sphere_grid(n_points) -> np.ndarray:    # golden sprial from stack overflow
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


def surface(n):    # from psi4.vdw_surface.py
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


def shrake_rupley(xyz: np.ndarray, atom_radii: np.ndarray, out_points: np.ndarray, n_points=0) -> np.ndarray:
    # prepare variables
    n_atoms = int(xyz.shape[0])
    neighbor_indices = np.zeros((n_atoms), dtype=float)
    # loop over all atoms
    atom_radii2 = atom_radii**2
    is_accessible = True
    # factor_packaging = (3.) / (4. * PI)
    # # max_rad + rad of spheres to accounbt for half spheres
    # max_rad = np.max(atom_radii) + 1.
    n_out_points = n_points
    for i in range(n_atoms):
        rad_i = atom_radii[i]
        centered_sphere_points = surface(int(4.0 * np.pi * atom_radii[i]**2))
        r_i = xyz[i, :]

        n_neighbor_indices = 0
        for j in range(n_atoms):
            if i == j:
                continue

            rad_cutoff2 = (rad_i + atom_radii[j])**2
            r2 = np.sum((r_i - xyz[j, :])**2, 0)
            if r2 < 1e-10:
                print("ERROR: THIS CODE IS KNOWN TO FAIL WHEN ATOMS ARE VIRTUALLY")
                print("ON TOP OF ONE ANOTHER. YOU SUPPLIED TWO ATOMS %f", np.sqrt(r2))
                print("APART. QUITTING NOW")
                raise ValueError
            elif r2 < rad_cutoff2:
                neighbor_indices[n_neighbor_indices] = j
                n_neighbor_indices += 1
        # heuristic norm: number of atoms that would fit inside a box with max cutoff for this atom
        # if all atoms have a bond length of 1 angstrom (spheres) and are packaged at 100% efficiency
        # n = factor_packaging * (max_rad + rad_i)**3
        # if n_neighbor_indices > n:
        #     print(n_neighbor_indices, n)
        #     continue
        # center the sphere points on atom i
        centered_sphere_points = rad_i * centered_sphere_points + r_i

        k_closest_neighbor = 0
        for j in range(centered_sphere_points.shape[0]):
            is_accessible = True
            r_j = centered_sphere_points[j, :]
            # iterate through the sphere points by cycling through them
            # in a circle, startin with k_closest_neighbor and the wrapping
            # around
            for k in range(k_closest_neighbor, n_neighbor_indices + k_closest_neighbor):
                k_prime = k % n_neighbor_indices
                index = int(neighbor_indices[k_prime])
                r2 = atom_radii2[index]
                r_jk2 = np.sum((r_j - xyz[index, :])**2, 0)

                if r_jk2 < r2:
                    k_closest_neighbor = k
                    is_accessible = False
                    break

            if is_accessible:
                out_points[n_out_points] = r_j
                n_out_points += 1

    return n_out_points


def mk_layers(xyz: np.ndarray, atom_radii, density=1, shells=[1.4, 1.6, 1.8, 2.0]) -> np.ndarray:
    n_points = int(
        4 * np.pi * density * (sum(map(lambda x: (x * 2.)**2, shells))) * xyz.shape[0]
    )    # surface density of 1: 4*pi*r^2 with r_max = 2. -> 16.*pi
    mk_layers_points = np.ndarray((n_points, 3), dtype=float)
    # potentially parallelizable! every layer is one process
    n_points = 0
    for y in shells:
        n_points = shrake_rupley(xyz, y * atom_radii, mk_layers_points, n_points)
    return mk_layers_points[:n_points, :]
