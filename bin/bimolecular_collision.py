#!/usr/bin/env python3

#******************************************
#
#    SHARC Program Suite
#
#    SHARC-MN Extension
#
#    Copyright (c) 2025 University of Vienna
#    Copyright (c) 2025 University of Minnesota
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
#******************************************

# This script is a wrapper for sampling initial conditions for bimolecular processes. 
# by Yinan Shu, Jan. 19, 2025. 

import os
import subprocess
from optparse import OptionParser
import random 
import datetime
import math
import sys
import numpy

from constants import CM_TO_HARTREE, HARTREE_TO_EV, U_TO_AMU, ANG_TO_BOHR, Boltzmann_Eh_K, NUMBERS, MASSES, ISOTOPES

DEBUG = False

version = '4.0'
versiondate = datetime.date(2025, 3, 14)

def write_xyz_coordinates(combined_data, output_xyz_file):
    """
    Writes the x, y, z coordinates for each initial condition to a separate file in the specified format.

    Args:
        combined_data (dict): Combined data structure.
        output_xyz_file (str): Path to the output XYZ file.
    """
    with open(output_xyz_file, "w") as f:
        for condition in combined_data["conditions"]:
            # Number of atoms
            num_atoms = len(condition["atoms"])
            f.write(f"{num_atoms}\n")
            f.write(f"{condition['index']}\n")
            for atom in condition["atoms"]:
                f.write(f"{atom['element']} {atom['x']/ANG_TO_BOHR:.6f} {atom['y']/ANG_TO_BOHR:.6f} {atom['z']/ANG_TO_BOHR:.6f}\n")
    print(f"XYZ coordinates written to {output_xyz_file}.")


def write_combined_data(combined_data, output_file):
    """
    Writes the combined data to a file in the same format as the initconds.

    Args:
        combined_data (dict): Combined data structure.
        output_file (str): Path to the output file.
    """
    with open(output_file, "w") as f:
        # Write the header
        header = combined_data["header"]
        f.write(f"Ninit     {header['Ninit']}\n")
        f.write(f"Natom     {header['Natom']}\n")
        f.write(f"Repr      {header['Repr']}\n")
        f.write(f"Temp      {header['Temp']:.10f}\n")
        f.write(f"Eref      {header['Eref']:.10f}\n")
        f.write(f"Eharm     {header['Eharm']:.10f}\n")
        f.write("\n")

        # Write the equilibrium geometry
        f.write("Equilibrium\n")
        for atom in combined_data["equilibrium"]:
            f.write(f"{atom['element']}   {atom['atomic_number']:.1f}   {atom['x']:.8f}   {atom['y']:.8f}   {atom['z']:.8f}   "
                    f"{atom['mass']:.8f}   {atom['vx']:.8f}   {atom['vy']:.8f}   {atom['vz']:.8f}\n")
        f.write("\n")

        # Write initial conditions
        for condition in combined_data["conditions"]:
            f.write(f"Index     {condition['index']}\n")
            f.write("Atoms\n")
            for atom in condition["atoms"]:
                f.write(f"{atom['element']}   {atom['atomic_number']:.1f}   {atom['x']:.8f}   {atom['y']:.8f}   {atom['z']:.8f}   "
                        f"{atom['mass']:.8f}   {atom['vx']:.8f}   {atom['vy']:.8f}   {atom['vz']:.8f}\n")
            f.write("States\n")
            for state, values in condition["states"].items():
                f.write(f"{state:<12} {values['value_au']:.12f} a.u.    {values['value_ev']:.12f} eV\n")
            f.write("\n")
    print(f"Combined data written to {output_file}.")


def combine_molecule_data(molecule1_data, molecule2_data, E_col):
    """
    Combines molecule1_data and molecule2_data into a single data structure for merged molecules.

    Args:
        molecule1_data (dict): Data structure for molecule 1.
        molecule2_data (dict): Data structure for molecule 2.

    Returns:
        dict: Combined data structure.
    """
    # Validate that both molecules have the same number of initial conditions
    if len(molecule1_data["conditions"]) != len(molecule2_data["conditions"]):
        raise ValueError("Molecule 1 and Molecule 2 must have the same number of initial conditions.")

    # Combine headers
    combined_data = {
        "header": {
            "Ninit": molecule1_data["header"]["Ninit"],
            "Natom": molecule1_data["header"]["Natom"] + molecule2_data["header"]["Natom"],
            "Repr": None,
            "Temp": 0.0,  # Can be updated based on requirements
            "Eref": 0.0,  # Can be updated based on requirements
            "Eharm": 0.0,  # Can be updated based on requirements
        },
        "equilibrium": molecule1_data["equilibrium"] + molecule2_data["equilibrium"],
        "conditions": []
    }

    # Combine conditions
    for cond1, cond2 in zip(molecule1_data["conditions"], molecule2_data["conditions"]):
        combined_condition = {
            "index": cond1["index"],
            "atoms": cond1["atoms"] + cond2["atoms"],  # Combine atoms
            "states": {
                "Ekin": {
                    "value_au": cond1["states"]["Ekin"]["value_au"] + cond2["states"]["Ekin"]["value_au"] + E_col,
                    "value_ev": cond1["states"]["Ekin"]["value_ev"] + cond2["states"]["Ekin"]["value_ev"] + E_col*27.211396132,
                },
                "Epot": {
                    "value_au": cond1["states"]["Epot"]["value_au"] + cond2["states"]["Epot"]["value_au"],
                    "value_ev": cond1["states"]["Epot"]["value_ev"] + cond2["states"]["Epot"]["value_ev"],
                },
                "Etot": {
                    "value_au": cond1["states"]["Etot"]["value_au"] + cond2["states"]["Etot"]["value_au"] + E_col,
                    "value_ev": cond1["states"]["Etot"]["value_ev"] + cond2["states"]["Etot"]["value_ev"] + E_col*27.211396132,
                },
            }
        }
        combined_data["conditions"].append(combined_condition)

    print("Molecule data combined successfully.")
    return combined_data


def add_collision_velocity(molecule2_data, E_col):
    """
    Adds a uniform velocity to each atom of molecule 2 in the -z direction based on the collision energy.

    Args:
        molecule2_data (dict): Data structure for molecule 2.
        E_col (float): Collision energy in the same units as the masses in molecule2_data.
    """
    # Compute the total mass of molecule 2
    total_mass = sum(atom["mass"] for atom in molecule2_data["equilibrium"])

    if total_mass == 0:
        raise ValueError("Total mass of molecule 2 is zero. Check molecule2_data for missing atom masses.")

    # Compute the velocity
    velocity = math.sqrt(2 * E_col / total_mass)

    print(f"Computed uniform velocity for molecule 2 atoms: {velocity:.6f} bohr/(atomic_time)")

    # Add velocity to each atom in each initial condition
    for condition in molecule2_data["conditions"]:
        for atom in condition["atoms"]:
            atom["vz"] -= velocity  # Add the velocity in the -z direction

    print("Collision velocity added to molecule 2 atoms.")


def random_rotation_matrix():
    """
    Generates a random 3D rotation matrix.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    # Generate a random quaternion
    q = numpy.random.normal(0, 1, 4)
    q /= numpy.linalg.norm(q)  # Normalize the quaternion

    # Extract quaternion components
    q0, q1, q2, q3 = q

    # Construct the rotation matrix
    R = numpy.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2)],
        [2 * (q1*q2 + q0*q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q0*q1)],
        [2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2)]
    ])
    return R


def rotate_molecule_with_random_rotation(molecule_data, if12, z_x_values):
    """
    Rotates the geometry of a molecule with a unique random rotation matrix for each initial condition,
    preserving the center of mass defined by z_x_values.

    Args:
        molecule_data (dict): Molecule data structure.
        z_x_values (list of tuples): List of (z, x) positions for each initial condition.
    """
    # Ensure the number of initial conditions matches the z_x_values length
    if if12==2:
        if len(molecule_data["conditions"]) != len(z_x_values):
            raise ValueError("Number of initial conditions must match the length of z_x_values.")
        # Rotate the geometry for each initial condition
        for condition, (z_cm, x_cm) in zip(molecule_data["conditions"], z_x_values):
            # Generate a unique random rotation matrix for this initial condition
            rotation_matrix = random_rotation_matrix()
            for atom in condition["atoms"]:
                # Translate atom to the center of mass
                position = numpy.array([atom["x"] - x_cm, atom["y"], atom["z"] - z_cm])
                # Apply the unique rotation for this condition
                rotated_position = rotation_matrix @ position
                # Translate atom back to the specified center of mass
                atom["x"], atom["y"], atom["z"] = rotated_position + numpy.array([x_cm, 0, z_cm])
    elif if12==1:
        for condition in molecule_data["conditions"]:
            rotation_matrix = random_rotation_matrix()
            for atom in condition["atoms"]:
                position = numpy.array([atom["x"], atom["y"], atom["z"]])
                rotated_position = rotation_matrix @ position
                atom["x"], atom["y"], atom["z"] = rotated_position


def generate_atom_data(n, atom, position, z_x_values=None):
    """
    Generates initial conditions for a single atom.

    Args:
        n (int): Number of initial conditions.
        atom (str): Symbol of the atom (e.g., 'H' for hydrogen).
        position (int): 1 for placing atom at origin, 2 for placing atom at z_x_values.
        z_x_values (list): List of tuples [(x, y), ...] specifying atom positions for each initial condition if position=2.

    Returns:
        dict: A data structure similar to molecule1_data/molecule2_data.
    """
    if atom not in MASSES:
        raise ValueError(f"Atom symbol '{atom}' is not in the MASSES dictionary.")

    if position == 2 and (z_x_values is None or len(z_x_values) != n):
        raise ValueError("z_x_values must be provided and match the number of initial conditions when position=2.")

    atom_mass = MASSES[atom]/U_TO_AMU
    atom_number = NUMBERS[atom]

    # Generate the atom data structure
    atom_data = {
        "header": {
            "Ninit": n,
            "Natom": 1,
            "Repr": None,
            "Temp": 0.0,
            "Eref": 0.0,
            "Eharm": 0.0,
        },
        "equilibrium": [
            {
                "element": atom,
                "atomic_number": float(atom_number),
                "x": 0.0 if position == 1 else z_x_values[0][1],
                "y": 0.0,
                "z": 0.0 if position == 1 else z_x_values[0][0],
                "mass": atom_mass,
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
            }
        ],
        "conditions": []
    }

    # Add initial conditions
    for i in range(n):
        atom_data["conditions"].append({
            "index": i + 1,
            "atoms": [
                {
                    "element": atom,
                    "atomic_number": float(atom_number),
                    "x": 0.0 if position == 1 else z_x_values[i][1],
                    "y": 0.0,
                    "z": 0.0 if position == 1 else z_x_values[i][0],
                    "mass": atom_mass,
                    "vx": 0.0,
                    "vy": 0.0,
                    "vz": 0.0,
                }
            ],
            "states": {
                "Ekin": {"value_au": 0.0, "value_ev": 0.0},
                "Epot": {"value_au": 0.0, "value_ev": 0.0},
                "Etot": {"value_au": 0.0, "value_ev": 0.0},
            },
        })

    return atom_data


def sample_z_x(n, bmin, bmax, strata, separation):
    """
    Samples z and x values for each initial condition.
    Args:
        n (int): Number of initial conditions.
        bmin (float): Minimum value of x.
        bmax (float): Maximum value of x.
        strata (int): Number of strata for x sampling.
        separation (float): Value of a, fixed for all initial conditions.
    Returns:
        list: A list of tuples (z,x) for each initial condition.
    """
    z = separation
    z_x_values = []

    for istrata in range(strata):
        indexmin = istrata * int(n / strata)
        indexmax = (istrata + 1) * int(n / strata)
        if istrata==strata-1:
            indexmax = n
        for i in range(indexmin,indexmax):
            bmin_stratum = bmin + istrata * (bmax - bmin) / strata
            bmax_stratum = bmin + (istrata + 1) * (bmax - bmin) / strata
            x = random.uniform((bmin_stratum*bmin_stratum), (bmax_stratum*bmax_stratum))
            z_x_values.append((z, math.sqrt(x)))

    return z_x_values


def move_to_origin(conditions):
    """
    Moves the center of mass of the molecule to the origin (0, 0, 0).
    Args:
        conditions (list): List of initial conditions, where each condition contains atom data.
    """
    for condition in conditions:
        current_com = compute_center_of_mass([condition])[0]
        shift_x = current_com["com_x"]
        shift_y = current_com["com_y"]
        shift_z = current_com["com_z"]

        for atom in condition["atoms"]:
            atom["x"] -= shift_x
            atom["y"] -= shift_y
            atom["z"] -= shift_z


def compute_center_of_mass(conditions):
    """
    Computes the center of mass for each initial condition in the conditions list.
    Args:
        conditions (list): List of initial conditions, where each condition contains atom data.
    Returns:
        list: List of dictionaries with COM for each condition.
    """
    com_results = []
    for condition in conditions:
        total_mass = 0.0
        com_x, com_y, com_z = 0.0, 0.0, 0.0
        for atom in condition["atoms"]:
            mass = atom["mass"]
            total_mass += mass
            com_x += mass * atom["x"]
            com_y += mass * atom["y"]
            com_z += mass * atom["z"]
        
        # Normalize by total mass
        com = {
            "index": condition["index"],
            "com_x": com_x / total_mass,
            "com_y": com_y / total_mass,
            "com_z": com_z / total_mass
        }
        com_results.append(com)
    
    return com_results


def move_to_z_x(conditions, z_x_values):
    """
    Moves the center of mass of the molecule to the specified (x, 0, z) for each initial condition.
    Args:
        conditions (list): List of initial conditions, where each condition contains atom data.
        z_x_values (list): List of (z, x) tuples for each initial condition.
    """
    for condition, (z, x) in zip(conditions, z_x_values):
        current_com = compute_center_of_mass([condition])[0]
        shift_x = current_com["com_x"] - x
        shift_y = current_com["com_y"]
        shift_z = current_com["com_z"] - z

        for atom in condition["atoms"]:
            atom["x"] -= shift_x
            atom["y"] -= shift_y
            atom["z"] -= shift_z


def parse_initconds_file(filename):
    """
    Parses Initial Conditions file and extracts the relevant information.
    Args:
        filename (str): Path to the initconds file (e.g., file from --o1 or --o2).
    Returns:
        dict: A dictionary containing parsed information with keys for global data and each index.
    """
    data = {"header": {}, "equilibrium": [], "conditions": []}
    with open(filename, "r") as f:
        lines = f.readlines()

    # Parse the header
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Ninit") or line.startswith("Natom") or line.startswith("Repr") or \
           line.startswith("Temp") or line.startswith("Eref") or line.startswith("Eharm"):
            key, value = line.split()
            data["header"][key] = float(value) if key != "Repr" else value
        elif line.startswith("Equilibrium"):
            i += 1
            while lines[i].strip():
                eq_line = lines[i].strip().split()
                data["equilibrium"].append({
                    "element": eq_line[0],
                    "atomic_number": float(eq_line[1]),
                    "x": float(eq_line[2]),
                    "y": float(eq_line[3]),
                    "z": float(eq_line[4]),
                    "mass": float(eq_line[5]),
                    "vx": float(eq_line[6]),
                    "vy": float(eq_line[7]),
                    "vz": float(eq_line[8]),
                })
                i += 1
        elif line.startswith("Index"):
            condition = {
                "index": int(line.split()[1]),
                "atoms": [],
                "states": {}
            }
            i += 1
            while i < len(lines) and lines[i].strip() != "States":
                atom_line = lines[i].strip().split()
                if len(atom_line) == 9:
                    condition["atoms"].append({
                        "element": atom_line[0],
                        "atomic_number": float(atom_line[1]),
                        "x": float(atom_line[2]),
                        "y": float(atom_line[3]),
                        "z": float(atom_line[4]),
                        "mass": float(atom_line[5]),
                        "vx": float(atom_line[6]),
                        "vy": float(atom_line[7]),
                        "vz": float(atom_line[8]),
                    })
                i += 1
            i += 1  # Skip "States"
            while i < len(lines) and lines[i].strip():
                state_line = lines[i].strip().split()
                condition["states"][state_line[0]] = {
                    "value_au": float(state_line[1]),
                    "value_ev": float(state_line[1])*HARTREE_TO_EV
                }
                i += 1
            data["conditions"].append(condition)
        i += 1


    return data


def read_initconds_files1(o):
    """
    Reads the initial condition files specified in --o1 and --o2.
    Args:
        o (str): File name for molecule 1 or 2.
    Returns:
        tuple: Parsed data for molecule 1 or 2..
    """
    if not os.path.isfile(o):
        raise FileNotFoundError(f"{o} not found.")

    print(f"Reading {o}...")
    molecule_data = parse_initconds_file(o)

    return molecule_data


def read_initconds_files(o1, o2):
    """
    Reads the initial condition files specified in --o1 and --o2.
    Args:
        o1 (str): File name for molecule 1.
        o2 (str): File name for molecule 2.
    Returns:
        tuple: Parsed data for molecule 1 and molecule 2.
    """
    if not os.path.isfile(o1):
        raise FileNotFoundError(f"{o1} not found.")
    if not os.path.isfile(o2):
        raise FileNotFoundError(f"{o2} not found.")

    print(f"Reading {o1}...")
    molecule1_data = parse_initconds_file(o1)
    print(f"Reading {o2}...")
    molecule2_data = parse_initconds_file(o2)

    return molecule1_data, molecule2_data


def delete_files(files):
    """
    Deletes the specified files.

    Args:
        files (list): List of file paths to delete.
    """
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted file: {file}")
        except FileNotFoundError:
            print(f"File not found, could not delete: {file}")
        except Exception as e:
            print(f"Error deleting file {file}: {e}")


def main():
  '''Main routine'''

  usage='''
bimolecular_collision.py [options] initconds1 initconds2

NOTICE:
    Our convention is using second molecule to collide towards the first molecule.

This script reads two initconds files and arranges them for molecular collision

Part of the code is adopted from wigner.py

Author: Yinan Shu
'''

  description=''

  parser = OptionParser(usage=usage, description=description)
  parser.add_option('-n', dest='n', type=int, nargs=1, default=3, help="Number of geometries to be generated (integer, default=3)")
  parser.add_option('--system', dest='system', type=str, nargs=1, default='3+3', help="type of the two systems, default is '3+3' which means its polyatomic molecules collides with polyatomic molecules, and '3' stands for polyatomic molecules with 3 or more atoms; options can be '1+2', '1+3', '2+1', '2+2', '2+3', '3+2', and '3+3', where 1 and 2 stand for atom and diatom respectively")
  parser.add_option('--atom', dest='atom', type=str, nargs=1, default='H', help="the atom involved in 1+2, 1+3, 2+1, and 3+1 systems")


  parser.add_option('--bmin', dest='bmin', type=float, nargs=1, default=0.0, help="minimal value of impact parameter in unit of angstrom")
  parser.add_option('--bmax', dest='bmax', type=float, nargs=1, default=5.0, help="maximum value of impact parameter in unit of angstrom")
  parser.add_option('--strata', dest='strata', type=int, nargs=1, default=1, help="number of strata between bmin and bmax, default value is 1, if a value larger than 1 is given, for example, --strata 3, then integer(n/3) initial conditions is given for each strata for the first 2 strata, and last strata has (n-2n/3) geometries. In addition, for ith strata, the bmin is bmin+(i-1)*(bmax-bmin), and bmax is bmin+i*(bmax-bmin)")
  parser.add_option('--separation', dest='separation', type=float, nargs=1, default=10.0, help="initial separation of the center of mass of two molecules, in unit of angstrom")
  parser.add_option('--relative_trans', dest='relative_trans',  type=float, nargs=1, default=1.0, help="initial relative translational energy between the center of mass of two molecules, in unit of eV")
  parser.add_option('--no_random_orient', dest='no_random_orient', action='store_true', help="Randomly orient the molecules in the space")


  parser.add_option('-o', dest='o', type=str, nargs=1, default='initconds', help="Output filename (string, default=""initconds"")")
  parser.add_option('-x', dest='X', action='store_true',help="Generate a xyz file with the sampled geometries in addition to the initconds file")

  (options, args) = parser.parse_args()

  system=options.system

  #==============
  # Sampling coordinates
  #==============
  # used for sampling coordinates 
  n=options.n
  bmin=options.bmin*ANG_TO_BOHR
  bmax=options.bmax*ANG_TO_BOHR
  strata=options.strata
  separation=options.separation*ANG_TO_BOHR
  #compute initial separation 
  z_x_values=sample_z_x(n, bmin, bmax, strata, separation)

  # first check how many inputs are given 
  if system=='2+3' or system=='3+2' or system=='3+3': 
      print("====================================================")
      print("Initial Condition Sampling for MOLECULE + MOLECULE")
      print("====================================================")
      if len(args) != 2:
          parser.error("You must provide exactly two input files (two initial condition files)")
      input_file1, input_file2 = args
      molecule1_data, molecule2_data = read_initconds_files(input_file1, input_file2)
      #move molecule 1 to origin 
      print("center of mass of MOLECULE 1 is moved to (0, 0, 0)")
      move_to_origin(molecule1_data["conditions"])
      #move molecule 2 to (x,y,0)
      print("center of mass of MOLECULE 2 is moved to (impact_parameter, 0, initial_separation)")
      move_to_z_x(molecule2_data["conditions"], z_x_values)

  elif system=='1+2' or system=='1+3': 
      print("====================================================")
      print("Initial Condition Sampling for ATOM + MOLECULE")
      print("====================================================")
      atom=options.atom
      print("Involved atom is:", atom)
      if len(args) !=1:
          parser.error("You must provide exactly one input file (one initial condition file)")
      input_file2 = args[0]
      # generate atom data (molecule 1)
      print("place atom at (0, 0, 0)")
      molecule1_data = generate_atom_data(n, atom, 1, z_x_values)
    #   print("perform initial condition sampling for MOLECULE 2 using state_selected.py")
      molecule2_data = read_initconds_files1(input_file2)
      # move molecule 2 to (x,y,0)
      print("center of mass of MOLECULE 2 is moved to (impact_parameter, 0, initial_separation)")
      move_to_z_x(molecule2_data["conditions"], z_x_values)

  elif system=='2+1' or system=='3+1':
      print("====================================================")
      print("Initial Condition Sampling for MOLECULE + ATOM")
      print("====================================================")
      atom=options.atom
      print("Involved atom is:", atom)
      if len(args) !=1:
          parser.error("You must provide exactly one input file (one initial condition file)")
      input_file1 = args[0]
      molecule1_data = read_initconds_files1(input_file1)
      # move molecule 1 to origin 
      print("center of mass of MOLECULE 1 is moved to (0, 0, 0)")
      move_to_origin(molecule1_data["conditions"])
      # generate atom data (molecule 2)
      print("place atom at (impact_parameter, 0, initial_separation)")
      molecule2_data = generate_atom_data(n, atom, 2, z_x_values)

  if not DEBUG:
      delete_files(['KEYSTROKES.bimolecular_collision'])

  #==============
  # Randomly orient the molecule
  #==============
  if not options.no_random_orient:
      # rotate the first molecule
      if system!='1+2' and system!='1+3':
          print("Random orient MOLECUL 1")
          rotate_molecule_with_random_rotation(molecule1_data, 1, z_x_values)
      # rotate the second molecule
      elif system!='2+1' and system!='3+1':
          print("Random orient MOLECUL 2")
          rotate_molecule_with_random_rotation(molecule2_data, 2, z_x_values)
            
  #==============
  # Sampling collision velocity
  #==============
  # add relative translation energy to molecule2
  E_col=options.relative_trans/HARTREE_TO_EV
  add_collision_velocity(molecule2_data, E_col)


  #==============
  # Put to initial condition together
  #==============
  combined_data=combine_molecule_data(molecule1_data, molecule2_data, E_col)
  outfile=options.o
  write_combined_data(combined_data, outfile)

  if options.X:
      write_xyz_coordinates(combined_data,options.o+'.xyz')

  # save the shell command
  command='python '+' '.join(sys.argv)
  f=open('KEYSTROKES.bimolecular_collision','w')
  f.write(command)
  f.close()

# ======================================================================================================================

if __name__ == '__main__':
    main()

