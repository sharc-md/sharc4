#!/usr/bin/env python3
import parmed as pmd
import re
import os
from constants import au2a


def expand_str_to_list(input: str) -> list[int]:
    out = []
    for i in input.split(","):
        if "~" in i:
            q = i.split("~")
            for j in range(int(q[0]), int(q[1]) + 1):
                out.append(j)
        else:
            out.append(int(i))
    return out


def main(file, qm_list, rattle_hx=False):
    qm_list = expand_str_to_list(qm_list)
    prmtop: pmd.amber.AmberParm = pmd.amber.LoadParm(file)
    natom = prmtop.ptr("NATOM")
    atom_symb = [re.match(r"[A-Z][a-z]?", prmtop.atoms[i].name)[0] for i in range(natom)]
    # types = prmtop._prmtop.getAtomTypes()
    bonds = prmtop.bonds
    atoms = prmtop.atoms
    atoms.index_members()
    atom_type = ["mm"] * natom
    for i in qm_list:
        atom_type[i - 1] = "qm"

    # construct all unique bonds

    # get bonds to rattle
    if rattle_hx:
        print("rattling H-X")
        with open("rattle", "w") as f:
            for bond in bonds:
                i, j = bond.atom1.idx, bond.atom2.idx
                if atom_symb[i] == "H" or atom_symb[j] == "H":
                    Rmin = bond.type.req
                    f.write(f"{i + 1: 6d} {j + 1: 6d} {Rmin / au2a: 10.7f}\n")

    # write QMMM.table from unique bonds
    with open("QMMM.table", "w") as f:
        for i, s in enumerate(atom_symb):
            f.write(f"{atom_type[i]}  {s:<4s} ")
            bonds_i = [b.atom2.idx for b in bonds if atoms[i] is b.atom1]
            f.write(" ".join(map(str, map(lambda x: x + 1, bonds_i))))
            f.write("\n")

    # set charges to Zero
    for i in qm_list:
        prmtop.atoms[i - 1].charge = 0.0

    filename = os.path.basename(file)
    filebase, ext = os.path.splitext(filename)
    prmtop.write_parm(filebase + "_chrg_0" + ext)

    # truncate the prmtop file
    atoms_to_keep = {i - 1 for i in qm_list}
    for b in bonds:
        if atom_type[b.atom1.idx] == "qm" and atom_type[b.atom2.idx] == "mm":
            atoms_to_keep.add(b.atom2.idx)
        elif atom_type[b.atom1.idx] == "mm" and atom_type[b.atom2.idx] == "qm":
            atoms_to_keep.add(b.atom1.idx)
    atoms_to_keep = sorted(atoms_to_keep)

    truncated_prmtop = prmtop[atoms_to_keep]
    truncated_prmtop.write_parm(filebase + "_qm_and_links_chrg0" + ext)


if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--prmtop", dest="file", help="Specify a prmtop file")
    parser.add_option(
        "-q",
        "--qm-list",
        type="str",
        default="",
        dest="qm_list",
        help="Specify 'QM' atoms as list starting from 1 (e.g. 1~3,5,8~12,20)\ndefault=\"\"",
    )
    parser.add_option(
        "--rattle-hx",
        action="store_true",
        default=False,
        dest="rattle_hx",
        help="Create RATTLE file for H-X bonds\ndefault=False",
    )

    (options, args) = parser.parse_args()
    if not options.file:
        parser.error("Filename not given")
    main(options.file, options.qm_list, options.rattle_hx)
