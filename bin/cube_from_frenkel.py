#!/usr/bin/env python3
import argparse
import pickle
from constants import IAn2AName, au2a

import numpy as np


def gaussian_density(
    q_alpha: np.ndarray, coords: np.ndarray, origin: np.ndarray, spacing: float, shape: tuple[int, int, int], sigma: float = 1.0
) -> np.ndarray:
    """
    rho(r) = sum_a q(a) * exp(-|r-Ra|^2 / (2 sigma^2))
    """
    nx, ny, nz = shape
    sigma2 = sigma**2
    pref = 1.0  # visualization scale

    # coordinate arrays
    xs = origin[0] + spacing * np.arange(nx)
    ys = origin[1] + spacing * np.arange(ny)
    zs = origin[2] + spacing * np.arange(nz)

    # allocate grid
    rho = np.zeros((nx, ny, nz), dtype=float)

    for qa, (xa, ya, za) in zip(q_alpha, coords):
        if qa == 0.0:
            continue
        dx2 = (xs - xa) ** 2
        dy2 = (ys - ya) ** 2
        dz2 = (zs - za) ** 2

        ex = np.exp(-dx2 / (2.0 * sigma2))
        ey = np.exp(-dy2 / (2.0 * sigma2))
        ez = np.exp(-dz2 / (2.0 * sigma2))

        # rho[ix,iy,iz] += qa * ex[ix]*ey[iy]*ez[iz]
        rho += pref * qa * (ex[:, None, None] * ey[None, :, None] * ez[None, None, :])

    return rho


def write_cube(filename: str, origin: np.ndarray, spacing: float, grid: np.ndarray, atom_z: np.ndarray, coords: np.ndarray):
    """
    Write a Gaussian cube file.

    filename:   Output path
    origin:     Cube origin
    spacing:    Grid spacing
    grid:       Grid points
    atom_z:     Atomic numbers
    coords:     Atom coordinates
    """
    nx, ny, nz = grid.shape
    ox, oy, oz = origin

    atom_z = np.asarray(atom_z)
    coords = np.asarray(coords, dtype=float)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("CUBE file\n\n")
        f.write(f"{len(atom_z):5d} {ox:12.6f} {oy:12.6f} {oz:12.6f}\n")
        f.write(f"{nx:5d} {spacing:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        f.write(f"{ny:5d} {0.0:12.6f} {spacing:12.6f} {0.0:12.6f}\n")
        f.write(f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {spacing:12.6f}\n")

        # Atom block: build all lines and write once
        atom_lines = [f"{int(Z):5d} {0.0:12.6f} {x:12.6f} {y:12.6f} {z:12.6f}\n" for Z, (x, y, z) in zip(atom_z, coords)]
        f.writelines(atom_lines)

        # Volumetric block
        flat = grid.reshape(-1)

        # Pad to multiple of n_values
        n = flat.size
        rem = n % 6
        if rem:
            pad = 6 - rem
            flat = np.pad(flat, (0, pad), mode="constant", constant_values=0.0)
        data2d = flat.reshape(-1, 6)

        np.savetxt(f, data2d, fmt=" %13.5e")


def make_grid(
    atom_coords_bohr: np.ndarray, padding_bohr: float = 6.0, spacing_bohr: float = 0.3
) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    """
    Make a rectangular grid around atoms.
    """
    mins = atom_coords_bohr.min(axis=0) - padding_bohr
    maxs = atom_coords_bohr.max(axis=0) + padding_bohr

    lengths = maxs - mins
    npts = np.ceil(lengths / spacing_bohr).astype(int) + 1
    nx, ny, nz = map(int, npts)

    origin = mins
    print(f"Gridsize: {npts*3}")
    return origin, float(spacing_bohr), (nx, ny, nz)


def pca_align_coords(coords_bohr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA-align a set of Cartesian coordinates.
    """
    coords = np.asarray(coords_bohr, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords_bohr must have shape (N,3)")

    n = coords.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for PCA alignment")

    w = np.ones(n, dtype=float)
    wsum = np.sum(w)
    if wsum <= 0:
        raise ValueError("Sum of weights must be > 0")

    # Weighted center
    center = (coords * w[:, None]).sum(axis=0) / wsum

    # Weighted covariance (equivalently PCA of centered points)
    X = coords - center
    Xw = X * np.sqrt(w)[:, None]
    cov = (Xw.T @ Xw) / wsum

    evals, evecs = np.linalg.eigh(cov)

    # Sort by descending variance (largest eigenvalue first)
    order = np.argsort(evals)[::-1]
    R = evecs[:, order]

    # Optional: enforce right-handed coordinate system
    if np.linalg.det(R) < 0.0:
        R[:, -1] *= -1.0

    coords_rot = X @ R
    return coords_rot, R, center


def write_xyz(filename: str, coords: np.ndarray, elements: list[str]):
    """
    Write xyz file
    """
    coords = np.asarray(coords, float)
    n = coords.shape[0]
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{n}\n\n")
        for el, (x, y, z) in zip(elements, coords):
            f.write(f"{el:2s} {x:14.6f} {y:14.6f} {z:14.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Resample and smooth Gaussian cube file.")
    parser.add_argument("cube_data", type=str, help="Input file")
    parser.add_argument("--spacing", type=float, default=0.3, help="New grid spacing (default: 0.3 Bohr)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma (default: 1.0 Bohr)")
    parser.add_argument("--padding", type=float, default=6.0, help="Grid padding (default: 6.0 Bohr)")
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
    write_xyz(f"{args.output}.xyz", coords_rot * au2a, [IAn2AName[a] for a in atom_charges])
    print(f"Generating grid with {args.padding} padding, {args.spacing} spacing...")
    origin, spacing, shape = make_grid(coords_rot, padding_bohr=args.padding, spacing_bohr=args.spacing)

    for s in states:
        rho = gaussian_density(charges[s], coords_rot, origin, spacing, shape, args.sigma)
        max_rho = np.max(np.abs(rho))
        print(f"Recommended isovalue for state {s:3d}: {0.02*max_rho:.6f} - {0.05*max_rho:.6f}")
        write_cube(f"{args.output}_{s:03d}.cube", origin, spacing, rho, atom_charges, coords_rot)


if __name__ == "__main__":
    main()
