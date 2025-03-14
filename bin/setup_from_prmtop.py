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


import parmed as pmd
import re
import os
from constants import au2a


def expand_str_to_list(input: str) -> list[int]:
    out = []
    for i in input.split():
        if "~" in i:
            q = i.split("~")
            for j in range(int(q[0]), int(q[1]) + 1):
                out.append(j)
        else:
            out.append(int(i))
    return out

# This is essentially expand_str_to_list() backwards
def bool_list_to_ranges(bool_list):
    # Identify the indices of True values
    indices = [i + 1 for i, value in enumerate(bool_list) if value=="qm"]

    # If there are no True values, return an empty string
    if not indices:
        return ""

    # Create ranges from the indices
    ranges = []
    start = indices[0]
    end = start

    for i in indices[1:]:
        if i == end + 1:  # Consecutive value
            end = i
        else:  # Non-consecutive, end the current range
            ranges.append((start, end))
            start = end = i
    ranges.append((start, end))  # Add the last range

    # Convert ranges into the desired string format
    range_strings = [f"{s}~{e}" if s != e else f"{s}" for s, e in ranges]
    return " ".join(range_strings)





def main(file, qm_list, rattle_hx=False, atommask=False):
    qm_list = expand_str_to_list(qm_list)
    if not qm_list:
        print("Please give a list of QM atoms with -q")
        exit()
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

    # get bonds to rattle
    if rattle_hx:
        print("rattling all H-X bonds ...")
        print("writing file 'rattle'")
        with open("rattle", "w") as f:
            for bond in bonds:
                i, j = bond.atom1.idx, bond.atom2.idx
                if atom_symb[i] == "H" or atom_symb[j] == "H":
                    Rmin = bond.type.req
                    f.write(f"{i + 1: 6d} {j + 1: 6d} {Rmin / au2a: 10.7f}\n")

    # get atom mask
    if atommask:
        ranges_string = bool_list_to_ranges(atom_type)
        print(f"making atom mask file ... (atom mask is '{ranges_string}')")
        print("writing file 'atommask'")
        string = ''
        for i in atom_type:
            if i == "qm":
                string += 'T\n'
            else:
                string += 'F\n'
        with open("atommask", "w") as f:
            f.write(string)

    # write QMMM.table from unique bonds
    print("writing file 'QMMM.table'")
    with open("QMMM.table", "w") as f:
        for i, s in enumerate(atom_symb):
            f.write(f"{atom_type[i]}  {s:<4s} ")
            bonds_i = [b.atom2.idx for b in bonds if atoms[i] is b.atom1]
            f.write(" ".join(map(str, map(lambda x: x + 1, bonds_i))))
            f.write("\n")

    # set charges to Zero
    print("setting QM charges to zero ...")
    for i in qm_list:
        prmtop.atoms[i - 1].charge = 0.0

    filename = os.path.basename(file)
    filebase, ext = os.path.splitext(filename)
    print(f"writing file '{filebase}_chrg_0{ext}'")
    prmtop.write_parm(filebase + "_chrg_0" + ext)

    # truncate the prmtop file
    print("truncating prmtop for QM-only ...")
    atoms_to_keep = {i - 1 for i in qm_list}
    for b in bonds:
        if atom_type[b.atom1.idx] == "qm" and atom_type[b.atom2.idx] == "mm":
            atoms_to_keep.add(b.atom2.idx)
        elif atom_type[b.atom1.idx] == "mm" and atom_type[b.atom2.idx] == "qm":
            atoms_to_keep.add(b.atom1.idx)
    atoms_to_keep = sorted(atoms_to_keep)

    truncated_prmtop = prmtop[atoms_to_keep]
    # in at least some cases, the box dimensions are zero, which leads to div-by-zero in QMMM interface 
    truncated_prmtop.parm_data["BOX_DIMENSIONS"] = prmtop.parm_data["BOX_DIMENSIONS"]  
    # set charges for link atoms to zero
    for i in range(len(truncated_prmtop.atoms)):
        truncated_prmtop.atoms[i].charge = 0.0
    print(f"writing file '{filebase}_qm_and_links_chrg0{ext}'")
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
    parser.add_option(
        "--atommask",
        action="store_true",
        default=False,
        dest="atommask",
        help="Create ATOMMASK file to exclude MM atoms from decoherence/rescaling\ndefault=False",
    )

    (options, args) = parser.parse_args()
    if not options.file:
        parser.error("Filename not given")
    main(options.file, options.qm_list, options.rattle_hx, options.atommask)
