#!/usr/bin/env python3
"""
Count contracted AO basis functions for an XYZ geometry using a PySCF basis set.

Default: counts from PySCF's internal basis data (no Mole object).
Optional: --use-mol builds a gto.Mole() and uses mol.nao_nr() for verification.

Examples
--------
python count_basis_pyscf.py water.xyz def2-svp
python count_basis_pyscf.py water.xyz cc-pvdz --cart
"""

import argparse
from collections import Counter

from pyscf import gto


def read_xyz_symbols(xyz_path: str) -> list[str]:
    """Read element symbols from an .xyz file."""
    syms = []
    with open(xyz_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            syms.append(parts[0])
    return syms


def cart_degeneracy(l: int) -> int:
    """# of Cartesian functions for angular momentum l."""
    return (l + 1) * (l + 2) // 2


def sph_degeneracy(l: int) -> int:
    """# of spherical harmonic functions for angular momentum l."""
    return 2 * l + 1


def count_from_pyscf_basis(symbols, basis_name, cart=False):
    """
    Count contracted AO basis functions using pyscf.gto.basis.load() without building a Mole.

    Counts contracted functions (not primitives). Uses spherical by default; use --cart for Cartesian.
    """
    deg = cart_degeneracy if cart else sph_degeneracy

    def shell_rows(shell):
        """
        Return (l, kappa, rows) where rows is an iterable of primitive rows.

        Handles common PySCF internal basis formats:
          A) [l, [exp,c..], [exp,c..], ...]
          B) [l, kappa, [exp,c..], [exp,c..], ...]
          C) [l, rows] where rows == [[exp,c..], ...]
          D) [l, exp1, c1, exp2, c2, ...]  (rare) -> convert to rows
          E) [l, exp1, [c1,c2..], exp2, [c1,c2..], ...] (rare) -> convert to rows
        """
        l = int(shell[0])
        kappa = 0

        if len(shell) < 2:
            return l, kappa, []

        # Case C: [l, rows] where rows is already a list of rows
        if isinstance(shell[1], list) and shell[1] and isinstance(shell[1][0], (list, tuple)):
            return l, kappa, shell[1]

        # Case B: [l, kappa, ...] (kappa is scalar int/float-like and next is not scalar-only shell content)
        start = 1
        if isinstance(shell[1], (int, float)) and len(shell) >= 3 and not isinstance(shell[2], (int, float, str)):
            kappa = int(shell[1])
            start = 2

        tail = shell[start:]

        # Case A: [l, row, row, ...] where each row is list/tuple
        if tail and isinstance(tail[0], (list, tuple)):
            return l, kappa, tail

        # Rare flattened variants: tail is scalars and/or scalar+list pairs
        # D: [exp1, c1, exp2, c2, ...] -> 1 contraction
        if tail and all(isinstance(x, (int, float)) for x in tail):
            if len(tail) % 2 != 0:
                raise ValueError(f"Unrecognized flattened shell format (odd length): {shell}")
            rows = [[float(tail[i]), float(tail[i + 1])] for i in range(0, len(tail), 2)]
            return l, kappa, rows

        # E: [exp1, [c...], exp2, [c...], ...]
        if tail and isinstance(tail[0], (int, float)) and len(tail) >= 2 and isinstance(tail[1], list):
            if len(tail) % 2 != 0:
                raise ValueError(f"Unrecognized exp/coeff shell format (odd length): {shell}")
            rows = []
            for i in range(0, len(tail), 2):
                expn = float(tail[i])
                coeffs = tail[i + 1]
                if not isinstance(coeffs, list):
                    raise ValueError(f"Unrecognized exp/coeff pairing in shell: {shell}")
                rows.append([expn] + [float(c) for c in coeffs])
            return l, kappa, rows

        raise ValueError(f"Unrecognized PySCF shell format: {shell}")

    per_element = Counter()
    total = 0
    sym_count = Counter(symbols)

    for sym, nat in sym_count.items():
        bas = gto.basis.load(basis_name, sym)
        nao_one_atom = 0

        for shell in bas:
            l, kappa, rows = shell_rows(shell)

            # This script targets normal (non-spinor) AO bases.
            if kappa not in (0,):
                raise ValueError(
                    f"Encountered kappa={kappa} for {sym}, l={l}. "
                    "This looks like a relativistic/spinor basis; counting differs."
                )

            if not rows:
                continue

            # Each row is [exp, c1, c2, ...]; number of contractions = (#columns - 1)
            first_row = rows[0]
            if not isinstance(first_row, (list, tuple)) or len(first_row) < 2:
                raise ValueError(f"Unexpected primitive row in {sym} shell l={l}: {first_row}")

            nctr = len(first_row) - 1
            nao_one_atom += nctr * deg(l)

        per_element[sym] = nao_one_atom * nat
        total += nao_one_atom * nat

    return total, per_element

def displaywelcome():
    string = '\n'
    string += '  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('RESP fit RAM size estimator') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Author: Sascha Mausenberger') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Version: 1.0') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    string += '''
Count contracted AO basis functions for an XYZ geometry using a PySCF basis set.
And calculate maximum amount of RAM usage based on block size.
  '''
    print(string)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", help="Path to .xyz file")
    ap.add_argument("basis", help="PySCF basis name (e.g., sto-3g, def2-svp, cc-pvdz)")
    ap.add_argument("--cart", action="store_true", help="Count Cartesian AOs instead of spherical")
    ap.add_argument("--block_size", type=int, default=5000, help="resp_block_size value")
    args = ap.parse_args()

    displaywelcome()

    symbols = read_xyz_symbols(args.xyz)
    if not symbols:
        raise SystemExit("No atoms parsed from XYZ (is the file valid?)")

    total, per_elem = count_from_pyscf_basis(symbols, args.basis, cart=args.cart)

    kind = "Cartesian" if args.cart else "Spherical"
    print(f"{kind} AO basis function count (from gto.basis.load): {total}")
    print("Breakdown by element:")
    for sym, n in sorted(per_elem.items()):
        print(f"  {sym:>2s}: {n}")
    print(f"\nMaximum block size: {args.block_size}")
    print(f"Estimated RAM usage: {total**2*args.block_size*8/1024**3:.2f} GB")


if __name__ == "__main__":
    main()
