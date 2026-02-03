#!/usr/bin/env python3
import argparse
import pickle
from typing import Tuple, Optional
from constants import IAn2AName, au2a

import numpy as np


def gaussian_smeared_density(
    q_alpha: np.ndarray,
    coords_bohr: np.ndarray,
    origin_bohr: np.ndarray,
    spacing_bohr: float,
    shape: Tuple[int, int, int],
    sigma_bohr: float = 1.0,
) -> np.ndarray:
    """
    rho(r) = sum_a q(a) * exp(-|r-Ra|^2 / (2 sigma^2))

    Notes:
    - This is not normalized; it's a visualization field.
    - Produces positive/negative lobes suitable for isosurfaces.
    """
    nx, ny, nz = shape
    sigma2 = float(sigma_bohr) ** 2
    pref = 1.0  # visualization scale; you can multiply later if you want

    # coordinate arrays
    xs = origin_bohr[0] + spacing_bohr * np.arange(nx)
    ys = origin_bohr[1] + spacing_bohr * np.arange(ny)
    zs = origin_bohr[2] + spacing_bohr * np.arange(nz)

    # allocate grid
    rho = np.zeros((nx, ny, nz), dtype=float)

    # Loop over atoms; vectorize over grid axes reasonably
    # For medium systems, this is fine. For huge systems, we can optimize further.
    for qa, (xa, ya, za) in zip(q_alpha, coords_bohr):
        if qa == 0.0:
            continue
        dx2 = (xs - xa) ** 2  # (nx,)
        dy2 = (ys - ya) ** 2  # (ny,)
        dz2 = (zs - za) ** 2  # (nz,)

        # separability: exp(-(dx^2+dy^2+dz^2)/2s^2)=ex*ey*ez
        ex = np.exp(-dx2 / (2.0 * sigma2))
        ey = np.exp(-dy2 / (2.0 * sigma2))
        ez = np.exp(-dz2 / (2.0 * sigma2))

        # outer products to form 3D contribution
        # rho[ix,iy,iz] += qa * ex[ix]*ey[iy]*ez[iz]
        rho += pref * qa * (ex[:, None, None] * ey[None, :, None] * ez[None, None, :])

    return rho


def softened_potential(
    q_alpha: np.ndarray,
    coords_bohr: np.ndarray,
    origin_bohr: np.ndarray,
    spacing_bohr: float,
    shape: Tuple[int, int, int],
    eta_bohr: float = 0.5,
) -> np.ndarray:
    """
    phi(r) = sum_a q(a) / sqrt(|r-Ra|^2 + eta^2)
    """
    nx, ny, nz = shape
    xs = origin_bohr[0] + spacing_bohr * np.arange(nx)
    ys = origin_bohr[1] + spacing_bohr * np.arange(ny)
    zs = origin_bohr[2] + spacing_bohr * np.arange(nz)

    phi = np.zeros((nx, ny, nz), dtype=float)
    eta2 = float(eta_bohr) ** 2

    # This is heavier than the Gaussian field (not separable).
    # Fine for moderate grids; for large ones consider coarser spacing.
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    for qa, (xa, ya, za) in zip(q_alpha, coords_bohr):
        if qa == 0.0:
            continue
        r2 = (X - xa) ** 2 + (Y - ya) ** 2 + (Z - za) ** 2
        phi += qa / np.sqrt(r2 + eta2)
    return phi


def write_cube(
    filename: str,
    origin_bohr: np.ndarray,
    spacing_bohr: float,
    grid: np.ndarray,
    atom_Z: np.ndarray,
    atom_coords_bohr: np.ndarray,
    comment1: str = "CUBE file",
    comment2: str = "Generated from exciton transition charges",
    values_per_line: int = 6,
):
    """
    Write a Gaussian cube file.

    Parameters
    ----------
    filename : str
        Output path.
    origin_bohr : (3,) array
        Cube origin in Bohr.
    spacing_bohr : float
        Grid spacing in Bohr (uniform in x,y,z).
    grid : (nx, ny, nz) array
        Scalar field values.
    atom_Z : (natoms,) array
        Atomic numbers.
    atom_coords_bohr : (natoms,3) array
        Atom coordinates in Bohr.
    """
    nx, ny, nz = grid.shape
    ox, oy, oz = origin_bohr

    atom_Z = np.asarray(atom_Z)
    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=float)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{comment1}\n")
        f.write(f"{comment2}\n")
        f.write(f"{len(atom_Z):5d} {ox:12.6f} {oy:12.6f} {oz:12.6f}\n")
        f.write(f"{nx:5d} {spacing_bohr:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        f.write(f"{ny:5d} {0.0:12.6f} {spacing_bohr:12.6f} {0.0:12.6f}\n")
        f.write(f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {spacing_bohr:12.6f}\n")

        # Atom block: build all lines and write once
        atom_lines = [
            f"{int(Z):5d} {0.0:12.6f} {x:12.6f} {y:12.6f} {z:12.6f}\n" for Z, (x, y, z) in zip(atom_Z, atom_coords_bohr)
        ]
        f.writelines(atom_lines)

        # Volumetric block
        flat = grid.reshape(-1)  # C-order: iz fastest, then iy, then ix (matches your loops)

        # Pad to multiple of values_per_line so reshape is safe
        n = flat.size
        rem = n % values_per_line
        if rem:
            pad = values_per_line - rem
            flat = np.pad(flat, (0, pad), mode="constant", constant_values=0.0)
        data2d = flat.reshape(-1, values_per_line)

        # np.savetxt writes one row per line. fmt controls spacing/format.
        # Use a leading space to mimic typical cube formatting.
        np.savetxt(f, data2d, fmt=" %13.5e")


def make_grid(
    atom_coords_bohr: np.ndarray,
    padding_bohr: float = 6.0,
    spacing_bohr: float = 0.3,
) -> Tuple[np.ndarray, float, Tuple[int, int, int]]:
    """
    Make a rectangular grid around atoms.

    Returns
    -------
    origin_bohr : (3,) array
    spacing_bohr : float
    shape : (nx, ny, nz)
    """
    mins = atom_coords_bohr.min(axis=0) - padding_bohr
    maxs = atom_coords_bohr.max(axis=0) + padding_bohr

    lengths = maxs - mins
    npts = np.ceil(lengths / spacing_bohr).astype(int) + 1
    nx, ny, nz = map(int, npts)

    origin = mins
    print(f"Gridsize: {npts*3}")
    return origin, float(spacing_bohr), (nx, ny, nz)


def pca_align_coords(
    coords_bohr: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    right_handed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA-align a set of Cartesian coordinates.

    Parameters
    ----------
    coords_bohr : (N,3) array
        Input coordinates (e.g., all atoms) in Bohr.
    weights : (N,) array, optional
        Optional weights (e.g., atomic masses). If None, uniform weights.
        Using masses makes this closer to an inertia-axis alignment.
    right_handed : bool
        If True, enforce det(R)=+1.

    Returns
    -------
    coords_rot : (N,3) array
        Rotated coordinates in the PCA frame.
    R : (3,3) array
        Rotation matrix such that coords_rot = (coords - center) @ R.
        Columns of R are principal axes.
    center : (3,) array
        Center used for alignment (weighted mean if weights provided).
    """
    coords = np.asarray(coords_bohr, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords_bohr must have shape (N,3)")

    n = coords.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for PCA alignment")

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError("weights must have shape (N,)")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

    wsum = np.sum(w)
    if wsum <= 0:
        raise ValueError("Sum of weights must be > 0")

    # Weighted center
    center = (coords * w[:, None]).sum(axis=0) / wsum

    # Weighted covariance (equivalently PCA of centered points)
    X = coords - center
    Xw = X * np.sqrt(w)[:, None]
    cov = (Xw.T @ Xw) / wsum  # (3,3) symmetric

    # Eigen-decomposition: eigh gives ascending eigenvalues
    evals, evecs = np.linalg.eigh(cov)

    # Sort by descending variance (largest eigenvalue first)
    order = np.argsort(evals)[::-1]
    R = evecs[:, order]

    # Optional: enforce right-handed coordinate system
    if right_handed and np.linalg.det(R) < 0.0:
        R[:, -1] *= -1.0

    coords_rot = X @ R
    return coords_rot, R, center


def write_xyz(filename: str, coords: np.ndarray, elements: list[str] | None = None, comment: str = ""):
    coords = np.asarray(coords, float)
    n = coords.shape[0]
    if elements is None:
        elements = ["X"] * n
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{n}\n{comment}\n")
        for el, (x, y, z) in zip(elements, coords):
            f.write(f"{el:2s} {x:14.6f} {y:14.6f} {z:14.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Resample and smooth Gaussian cube file.")
    parser.add_argument("cube_data", type=str, help="Input file")
    parser.add_argument("--spacing", type=float, default=0.3, help="New grid spacing (default: 0.3)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma (default: 1.0)")
    parser.add_argument("--padding", type=float, default=6.0, help="Grid padding (default: 6.0)")
    parser.add_argument("-o", "--output", type=str, default="output", help="Output cube file")
    parser.add_argument("-s", "--states", type=int, nargs="+", default=None, help="State indices")

    args = parser.parse_args()

    print(f"Parsing file: {args.cube_data}")
    with open(args.cube_data, "rb") as f:
        charges, coords, atom_charges = pickle.load(f)

    states = args.states
    if states is None:
        states = list(range(1, charges.shape[0]))
    print(f"States to generate: {states}")

    coords_rot, _, _ = pca_align_coords(coords)
    write_xyz(f"{args.output}.xyz", coords_rot*au2a, [IAn2AName[a] for a in atom_charges])
    print(f"Generating grid with {args.padding} padding, {args.spacing} spacing...")
    origin, spacing, shape = make_grid(coords_rot, padding_bohr=args.padding, spacing_bohr=args.spacing)

    for s in states:
        rho = gaussian_smeared_density(charges[s], coords_rot, origin, spacing, shape, args.sigma)
        max_rho = np.max(np.abs(rho))
        print(f"Recommended isovalue for state {s:3d}: {0.02*max_rho:.6f} - {0.05*max_rho:.6f}")
        write_cube(f"{args.output}_{s:03d}.cube", origin, spacing, rho, atom_charges, coords_rot)


if __name__ == "__main__":
    main()
