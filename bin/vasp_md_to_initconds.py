#!/usr/bin/env python

import os, sys, argparse
import numpy as np
import re
from constants import ANG_TO_BOHR,ATOMCHARGE,MASSES,au2fs,U_TO_AMU
from py4vasp import Calculation
import random
import mdtraj as md
from sklearn.cluster import AgglomerativeClustering

def random_initcond(path,Ninit,Nframes,movie):
    '''
    Generating initial conditions for SHARC by random sampling of the last specified frames of a VASP MD trajectory.
    '''
    NM_TO_BOHR=ANG_TO_BOHR*10
    calc=Calculation.from_path(path)
    traj=calc.structure[-Nframes:].to_mdtraj()
    traj.superpose(traj[0])  # align to first frame to remove translation
    data=calc.velocity[-Nframes:].read()
    veloc=data["velocities"]*ANG_TO_BOHR*au2fs #Because VASP velocities are in Ang./fs. We need atomic units.
    lattice_vectors=data["structure"]["lattice_vectors"] #Lattice vectors. Do not change with NVT simulations.
    elements=data["structure"]["elements"] #List of atom labels
    xyz=traj.xyz*NM_TO_BOHR #Coordinates in Bohr upon MDTraj reading.
    index=random.sample(range(0, len(xyz)), Ninit) #Random indexes for sampling Ninit initial conditions over last Nframes of the trajectory
    veloc_sample=veloc[index] #Sampled velocities
    xyz_sample=xyz[index] #Sampled coordinates
    Natoms=len(elements) #N. of atoms for the system.
    with open("initconds", "w") as f:
        f.write("SHARC Initial conditions file, version 4.0\n")
        f.write(f"Ninit     {Ninit:d}\n")
        f.write(f"Natoms     {Natoms:d}\n")
        f.write(f"Repr      None\n")
        f.write(f"Eref      {0.:18.10f}\n")
        f.write(f"Eharm      {0.:18.10f}\n")
        f.write("\n")
        f.write("Equilibrium\n")
        for i in range(Natoms):
            f.write(f" {elements[i]}   {ATOMCHARGE[elements[i]]:>.1f}  {xyz[0][i,0]:>12.8f} {xyz[0][i,1]:>12.8f} {xyz[0][i,2]:>12.8f} "\
                    f" {MASSES[elements[i]]/U_TO_AMU:>12.8f} {veloc[0][i,0]:>12.8f} {veloc[0][i,1]:>12.8f} {veloc[0][i,2]:>12.8f}\n")
        for n,i in enumerate(range(Ninit)):
            f.write("\n\n")
            f.write(f"Index    {i+1:>d}\n")
            f.write("Atoms\n")
            for j in range(Natoms):
                f.write(f" {elements[j]}   {ATOMCHARGE[elements[j]]:>.1f}  {xyz_sample[n][j,0]:>12.8f} {xyz_sample[n][j,1]:>12.8f} {xyz_sample[n][j,2]:>12.8f} "\
                        f" {MASSES[elements[j]]/U_TO_AMU:>12.8f} {veloc_sample[n][j,0]:>12.8f} {veloc_sample[n][j,1]:>12.8f} {veloc_sample[n][j,2]:>12.8f}\n")
            f.write(f"Ekin     {0.:>16.12f} a.u.\n")
            f.write(f"Epot_harm     {0.:>16.12f} a.u.\n")
            f.write(f"Epot     {0.:>16.12f} a.u.\n")
            f.write(f"Etot_harm     {0.:>16.12f} a.u.\n")
            f.write(f"Etot     {0.:>16.12f} a.u.\n")
        f.write("\n\n")
    
    if movie:
        make_dyn_file(elements,xyz_sample)
    return

def cluster_initcond(path,Ninit,Nframes,threshold,movie):
    '''
    Generating initial conditions for SHARC by RMSD-based cluster analysis of the last specified frames of a VASP MD trajectory.
    '''
    NM_TO_BOHR=ANG_TO_BOHR*10
    ANG_TO_NM=10
    calc=Calculation.from_path(path)
    traj=calc.structure[-Nframes:].to_mdtraj()
    traj.superpose(traj[0])  # align to first frame to remove translation
    data=calc.velocity[-Nframes:].read()
    veloc=data["velocities"]*ANG_TO_BOHR*au2fs #Because VASP velocities are in Ang./fs. We need atomic units.
    lattice_vectors=data["structure"]["lattice_vectors"] #Lattice vectors. Do not change with NVT simulations.
    elements=data["structure"]["elements"] #List of atom labels
    xyz=traj.xyz*NM_TO_BOHR #Coordinates in Bohr upon MDTraj reading.
    #Cluster analysis
    #Compute pairwise RMSD matrix ---
    print("Computing pairwise RMSD matrix...")
    rmsd_matrix = np.empty((traj.n_frames, traj.n_frames))
    for i in range(traj.n_frames):
        rmsd_matrix[i] = md.rmsd(traj, traj, i)  # RMSD between all frames and frame i
    # Symmetrize and set diagonal to zero
    rmsd_matrix = 0.5 * (rmsd_matrix + rmsd_matrix.T)
    np.fill_diagonal(rmsd_matrix, 0.0)
    # Cluster based on RMSD distances ---
    # Smaller threshold = more clusters, larger = fewer clusters
    print(f"Clustering...(selected cluster threshold {threshold} Angstrom)")
    clustering = AgglomerativeClustering(
        n_clusters=None,                # let distance threshold decide number of clusters
        metric='precomputed',
        linkage='average',
        distance_threshold=threshold*ANG_TO_NM          
    )
    labels = clustering.fit_predict(rmsd_matrix)
    # Find most populated cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    most_pop_cluster = sorted_clusters[0][0]
    frames_in_cluster = np.where(labels == most_pop_cluster)[0]
    print(f"Most populated cluster has {len(frames_in_cluster)} structures")
    #Pick Ninit representative structures from this cluster ---
    if len(frames_in_cluster) < Ninit:
        rep_indices = frames_in_cluster
        print(f"In the most representative cluster there are less than {Ninit} structures, so I will keep only those.")
    else:
        rep_indices = np.linspace(0, len(frames_in_cluster) - 1, Ninit, dtype=int)
        rep_indices = frames_in_cluster[rep_indices]
        print(f"{Ninit} are selected")
    #Saving sampled structures for initconds
    index=rep_indices 
    veloc_sample=veloc[index] #Sampled velocities
    xyz_sample=xyz[index] #Sampled coordinates
    Natoms=len(elements) #N. of atoms for the system.
    with open("initconds", "w") as f:
        f.write("SHARC Initial conditions file, version 4.0\n")
        f.write(f"Ninit     {Ninit:d}\n")
        f.write(f"Natoms     {Natoms:d}\n")
        f.write(f"Repr      None\n")
        f.write(f"Eref      {0.:18.10f}\n")
        f.write(f"Eharm      {0.:18.10f}\n")
        f.write("\n")
        f.write("Equilibrium\n")
        for i in range(Natoms):
            f.write(f" {elements[i]}   {ATOMCHARGE[elements[i]]:>.1f}  {xyz[0][i,0]:>12.8f} {xyz[0][i,1]:>12.8f} {xyz[0][i,2]:>12.8f} "\
                    f" {MASSES[elements[i]]/U_TO_AMU:>12.8f} {veloc[0][i,0]:>12.8f} {veloc[0][i,1]:>12.8f} {veloc[0][i,2]:>12.8f}\n")
        for n,i in enumerate(range(Ninit)):
            f.write("\n\n")
            f.write(f"Index    {i+1:>d}\n")
            f.write("Atoms\n")
            for j in range(Natoms):
                f.write(f" {elements[j]}   {ATOMCHARGE[elements[j]]:>.1f}  {xyz_sample[n][j,0]:>12.8f} {xyz_sample[n][j,1]:>12.8f} {xyz_sample[n][j,2]:>12.8f} "\
                        f" {MASSES[elements[j]]/U_TO_AMU:>12.8f} {veloc_sample[n][j,0]:>12.8f} {veloc_sample[n][j,1]:>12.8f} {veloc_sample[n][j,2]:>12.8f}\n")
            f.write(f"Ekin     {0.:>16.12f} a.u.\n")
            f.write(f"Epot_harm     {0.:>16.12f} a.u.\n")
            f.write(f"Epot     {0.:>16.12f} a.u.\n")
            f.write(f"Etot_harm     {0.:>16.12f} a.u.\n")
            f.write(f"Etot     {0.:>16.12f} a.u.\n")
        f.write("\n\n")
    
    if movie:
        make_dyn_file(elements,xyz_sample)
    return

def make_dyn_file(elements,xyz):
    fl = open("initconds.xyz", 'w')
    string=''
    for n, coords in enumerate(xyz):
        ICOND=f"ICOND {n+1}"
        string += '%i\n%s\n' % (len(elements), ICOND)
        for a,atom in enumerate(elements):
            string += '%s' % (atom)
            for j in range(3):
                string += ' %f' % (coords[a,j] / ANG_TO_BOHR)
            string += '\n'
    fl.write(string)
    fl.close()

def parse_cml_args(cml):
    '''
    command line parser.
    '''
    description='''
    This script generate a set of initial conditions (initconds file) for SHARC-VASP dynamics reading a MD trajectory computed with VASP.
    It is supposed to analyze the last user-specified Nframes of your trajectory where equilibration etc. is achieved.
    py4vasp and mdtraj python packages have to be installed in the user's python environment. 
    Two options are supported for
    generating initial conditions:
    1) Random sampling of n initial conditions from the last N frames specified by the user (--random).
    2) Sampling n initial conditions from most populated clusters upon cluster analysis (--cluster).
    '''
    arg = argparse.ArgumentParser(add_help=True)
   
    arg.add_argument(dest="path", 
                     help='Location of VASP folder where MD run was executed')

    arg.add_argument('-f', dest='Nframes', action='store', type=int, default=100, 
                     help='N. of last frames to read from trajectory.')
    
    arg.add_argument('-n', dest='Ninit', action='store', type=int, default=10,
                     help='N. of initial conditions to generate')
    
    arg.add_argument('--random', dest='random', action='store_true', 
                     help='Select randomly n initial conditions from the specified frames')
    
    arg.add_argument('--cluster', dest='cluster', action='store_true', 
                     help='Select n initial conditions from the specified frames upon cluster analysis of those.')
    
    arg.add_argument('--thold', dest='thold', action='store', type=float, default=1, 
                     help='RMSD threshold value for clustering. Default is 1 Ang.')
    
    arg.add_argument('-x', dest='X', action='store_true', 
                     help='Generate a xyz file with the sampled geometries in addition to the initconds file')
     
    return arg.parse_args(cml)

def main(cml):
    arg = parse_cml_args(cml)
    if arg.random:
        random_initcond(arg.path,arg.Ninit,arg.Nframes,arg.X)
    elif arg.cluster:
        cluster_initcond(arg.path,arg.Ninit,arg.Nframes,arg.thold,arg.X)

if __name__ == "__main__":
    main(sys.argv[1:])
