#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------- User Parameters --------------------
h5_file = "sim_file.hdf5"   # your HDF5 file
output_dir = "slices_pngs"
nth_step = 10                # plot every nth timestep
pml_thickness = 0.5e-6       # in meters (from simulation)
coordinates = "CARTESIAN"   # "CARTESIAN" or "CYLINDRICAL"
dpi = 150

os.makedirs(output_dir, exist_ok=True)

# -------------------- Load HDF5 Data --------------------
with h5py.File(h5_file, "r") as f:
    Ex = f["e_x_data_si"][:]  # shape: (time, x, y, z)
    Ey = f["e_y_data_si"][:]
    Ez = f["e_z_data_si"][:]
    Bx = f["b_x_data_si"][:]
    By = f["b_y_data_si"][:]
    Bz = f["b_z_data_si"][:]
    
    tmax = f.attrs["tmax_si"]
    dt_saved = f.attrs["saved_dt_si"]
    
    # axes info (corrected)
    x0, x1, dx = f.attrs["rxmin_si_output"], f.attrs["rxmax_si_output"], f.attrs["drx_si_output"]
    z0, z1, dz = f.attrs["zmin_si_output"], f.attrs["zmax_si_output"], f.attrs["dz_si_output"]
    
    if coordinates == "CARTESIAN":
        y0, y1, dy = f.attrs["ymin_si_output"], f.attrs["ymax_si_output"], f.attrs["dy_si_output"]

# Build physical axes
x_arr = np.linspace(x0, x1, Ex.shape[1]) * 1e6  # µm
z_arr = np.linspace(z0, z1, Ex.shape[3]) * 1e6  # µm
if coordinates == "CARTESIAN":
    y_arr = np.linspace(y0, y1, Ex.shape[2]) * 1e6

# -------------------- Precompute color bar limits --------------------
E2_max = 0
B2_max = 0
for t_idx in range(0, Ex.shape[0], nth_step):
    y_center_idx = Ex.shape[2] // 2
    Ex_slice = Ex[t_idx, :, y_center_idx, :]
    Ey_slice = Ey[t_idx, :, y_center_idx, :]
    Ez_slice = Ez[t_idx, :, y_center_idx, :]
    Bx_slice = Bx[t_idx, :, y_center_idx, :]
    By_slice = By[t_idx, :, y_center_idx, :]
    Bz_slice = Bz[t_idx, :, y_center_idx, :]
    
    E2 = Ex_slice**2 + Ey_slice**2 + Ez_slice**2
    B2 = Bx_slice**2 + By_slice**2 + Bz_slice**2
    
    E2_max = max(E2_max, E2.max())
    B2_max = max(B2_max, B2.max())

print(f"Max |E|^2: {E2_max:.3e}, Max |B|^2: {B2_max:.3e}")

# -------------------- Loop over timesteps --------------------
for t_idx in range(0, Ex.shape[0], nth_step):
    y_center_idx = Ex.shape[2] // 2
    Ex_slice = Ex[t_idx, :, y_center_idx, :]
    Ey_slice = Ey[t_idx, :, y_center_idx, :]
    Ez_slice = Ez[t_idx, :, y_center_idx, :]
    Bx_slice = Bx[t_idx, :, y_center_idx, :]
    By_slice = By[t_idx, :, y_center_idx, :]
    Bz_slice = Bz[t_idx, :, y_center_idx, :]
    
    # Total squared fields
    E2 = Ex_slice**2 + Ey_slice**2 + Ez_slice**2
    B2 = Bx_slice**2 + By_slice**2 + Bz_slice**2
    


    # Plot E-field magnitude
    plt.figure(figsize=(8,6))
    plt.imshow(E2.T, extent=[x_arr[0], x_arr[-1], z_arr[0], z_arr[-1]],
               origin='lower', aspect='auto', cmap='inferno', vmin=0, vmax=E2_max)
    plt.colorbar(label=r"$|E|^2$ (V/m)$^2$")
    plt.xlabel("x (µm)")
    plt.ylabel("z (µm)")
    plt.title(f"E-field magnitude, t={t_idx*dt_saved*1e15:.2f} fs")
    
    # PML regions
    plt.axvspan(x_arr[0], x_arr[0]+pml_thickness*1e6, color='cyan', alpha=0.3)
    plt.axvspan(x_arr[-1]-pml_thickness*1e6, x_arr[-1], color='cyan', alpha=0.3)
    plt.axhspan(z_arr[0], z_arr[0]+pml_thickness*1e6, color='cyan', alpha=0.3)
    plt.axhspan(z_arr[-1]-pml_thickness*1e6, z_arr[-1], color='cyan', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/E2_t{t_idx:05d}.png", dpi=dpi)
    plt.close()
    


    # Plot B-field magnitude
    plt.figure(figsize=(8,6))
    plt.imshow(B2.T, extent=[x_arr[0], x_arr[-1], z_arr[0], z_arr[-1]],
               origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=B2_max)
    plt.colorbar(label=r"$|B|^2$ (T)$^2$")
    plt.xlabel("x (µm)")
    plt.ylabel("z (µm)")
    plt.title(f"B-field magnitude, t={t_idx*dt_saved*1e15:.2f} fs")
    
    # PML regions
    plt.axvspan(x_arr[0], x_arr[0]+pml_thickness*1e6, color='cyan', alpha=0.3)
    plt.axvspan(x_arr[-1]-pml_thickness*1e6, x_arr[-1], color='cyan', alpha=0.3)
    plt.axhspan(z_arr[0], z_arr[0]+pml_thickness*1e6, color='cyan', alpha=0.3)
    plt.axhspan(z_arr[-1]-pml_thickness*1e6, z_arr[-1], color='cyan', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/B2_t{t_idx:05d}.png", dpi=dpi)
    plt.close()

print(f"Saved E2 and B2 slices every {nth_step} steps in {output_dir}")