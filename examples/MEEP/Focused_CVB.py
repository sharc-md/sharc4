#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 00:14:19 2020

@author: Gonzague Agez, Martin Montagnac, Arnaud Arbouet, Vincent Paillard
@Affiliation : CEMES - CNRS (France)

This script proposes two methods to implement Focused Cylindrical Vector Beams (Azimuthal and Radial) in meep

"""


# ===================== Imports =====================

# general
import numpy as np
import argparse
import time
import scipy.constants as const
import os
import bisect
import h5py

# MEEP specific
import meep as mp

# MEEP materials
# from meep.materials import fused_quartz #Au, Al, Pt, Ag 



# ===================== Input options =====================

start_time = time.time()
parser = argparse.ArgumentParser()  

# Simulation box parameters
parser.add_argument('-coordinates',  type=str,     default = 'CARTESIAN',   help='CARTESIAN/CYLINDRICAL')
parser.add_argument('-resolution',  type=int,     default = 16,   help='Spatial resolution (points per µm)')
parser.add_argument('-transversal_extension_1',         type=float,   default =  4.5,    help='in µm, transversal axis 1 from -sx to sx')
parser.add_argument('-transversal_extension_2',         type=float,   default =  4.5,    help='in µm, transversal axis 2 from -sy to sy') 
parser.add_argument('-sz',          type=float,   default = 5.5,    help='in µm, longitudinal size of the cell (without PML), .5 because then resolution*sz is odd')
parser.add_argument('-dpml',        type=float,   default = 0.5,    help='PML thickness will be set later as half of the wavelength')

# Simulation time parameters
parser.add_argument('-temp_stepsize',  type=float,  default = 5E-18,   help='temporal stepsize (sec)')
parser.add_argument('-runtime',     type=float,   default = 40.0,   help='in fs, run until')

# Beam parameters
parser.add_argument('-beam',        type=str,     default ='azi',  help='beam type : azi, rad')
parser.add_argument('-wvl',         type=float,   default = 527.5,     help='in nm, wavelength')
parser.add_argument('-w0',          type=float,   default = 2.5 ,  help='in µm, waist of the LG beam')

# Pulse parameters
parser.add_argument('-t_pulse_start',         type=float,   default = 0.0 ,    help='Pulse start time (fs)')
parser.add_argument('-t_pulse_rise',          type=float,   default = 5.0 ,    help='Pulse rise time (fs)')
parser.add_argument('-t_pulse_flat',          type=float,   default = 0.0 ,    help='Pulse flat time (fs)')
parser.add_argument('-t_pulse_fall',          type=float,   default = 5.0 ,    help='Pulse fall time (fs)')
parser.add_argument('-j_amp',                 type=float,   default = 2.7674596515882675,    help='Amplitude factor for E-field (default equates to 1.2GV/m in free space)')                       

# Output parameters
parser.add_argument('-readout_fac',  type=int,  default = 40,   help='Save every ith timestep')
parser.add_argument('-save_dir',    type=str,   default = os.getcwd(),      help='Directory for storage of output files')


# PARABOLA
parser.add_argument('-use_parabola', type=bool, default=False, help='Enable metallic parabola')
parser.add_argument('-rad_parabola_left_metal',   type=float,   default = 2.5/np.sqrt(2),    help='in µm, radial size of the parabola (without PML)')   
parser.add_argument('-rad_parabola_right_metal',   type=float,   default = 1.2,    help='in µm, radial size of the parabola (3.768)')                  
parser.add_argument('-delta_parabola_metal',   type=float,   default = 0.0,  help='in µm, longitudinal size of the parabola (3.768)') 
parser.add_argument('-len_metal',   type=float,   default = 0.8,  help='in µm, longitudinal size of the cylindric (3.768)')

# PARTICLE 
parser.add_argument('-use_particle', type=bool, default=False, help='Enable dielectric nanoparticle')
parser.add_argument('-d_particle',   type=float,   default = 0.06,  help='in µm, diameter of the nanoparticle')
parser.add_argument('-offset_part', type=float, default=0.0, help='Offset position from entrance of aperture')

args = parser.parse_args()
print("Simulation parameters:")
for k,v in vars(args).items():
    print(f"{k}: {v}")
print('-----------------------\n')





# ===================== Calculation setup =====================

# Coordinate parameter conversion
coordinates = args.coordinates
if coordinates=="CYLINDRICAL":
    raise NotImplementedError("Cylindrical coordinate system currently not supported by this script")
if coordinates=="CYLINDRICAL":
    sr = args.transversal_extension_1/2.
elif coordinates=="CARTESIAN":
    sx = args.transversal_extension_1
    sy = args.transversal_extension_2
sz          = args.sz
resolution  = args.resolution
beam        = args.beam
wvl         = args.wvl/1000 # convert to µm
dpml        = args.dpml
temp_stepsize = args.temp_stepsize
readout_fac = args.readout_fac
save_dir    = args.save_dir


# Pulse and beam parameter conversion
t0 = 1E-6/const.c 
runtime       = args.runtime * 1E-15 / t0
t_pulse_start = args.t_pulse_start * 1E-15 / t0 
t_pulse_rise  = args.t_pulse_rise  * 1E-15 / t0 
t_pulse_flat  = args.t_pulse_flat  * 1E-15 / t0 
t_pulse_fall  = args.t_pulse_fall  * 1E-15 / t0 
print(f"t_pulse_start: {t_pulse_start}")
print(f"t_pulse_rise: {t_pulse_rise}")
print(f"t_pulse_flat: {t_pulse_flat}")
print(f"t_pulse_fall: {t_pulse_fall}")
print(f"runtime: {runtime}")
j_amp        = args.j_amp
e_conv_fac = 1/(1E-6*const.epsilon_0*const.c)
b_conv_fac = 1/(1E-6*const.epsilon_0*const.c**2)
w0          = args.w0
fcen        = 1/wvl

# Parameters for the simulation box content

use_parabola = args.use_parabola
use_particle = args.use_particle

# VACUUM 
if not use_parabola and not use_particle:
    len_metal = 0
    delta_metal = 0
    print(f"len_metal (µm): {len_metal}")

# # PARABOLA 
# if use_parabola:
#     rad_parabola_left_metal = args.rad_parabola_left_metal                                                                                                  
#     rad_parabola_right_metal = args.rad_parabola_right_metal                                                                                                
#     len_metal = args.len_metal
#     delta_metal = args.delta_parabola_metal
#     focal_len = (rad_parabola_left_metal**2-rad_parabola_right_metal**2)/(4*len_metal)
#     base_point = rad_parabola_left_metal**2/(4*focal_len)
#     print(f"rad_parabola_left_metal (µm): {rad_parabola_left_metal}")
#     print(f"rad_parabola_right_metal (µm): {rad_parabola_right_metal}")
#     print(f"delta_metal (µm): {delta_metal}")
#     print(f"focal_len (µm): {focal_len}")
#     print(f"base_point (µm): {base_point}")

# # Particle parameters
# if use_particle:
#     d_particle = args.d_particle
#     offset_part = args.offset_part
#     print(f"d_particle (µm): {d_particle}")
#     print(f"offset_part (µm): {offset_part}")


# Simulation box setup
cell_z = sz+2*dpml 
if coordinates=="CYLINDRICAL":
    cell_r = sr+dpml
    cell_phi = 0
    cell_size = mp.Vector3(cell_r, cell_phi, cell_z)
    cell_center_r, cell_center_phi, cell_center_z = 0, 0, 0  
    cell_center = mp.Vector3(cell_center_r, cell_center_phi, cell_center_z)
    dimensions = mp.CYLINDRICAL
    # SOURCE PROPERTIES
    center_r = cell_r/2.
    center_phi = 0
    center_z = -sz/2.
    source_size = mp.Vector3(cell_r, 0.0,  0.0) 
    source_pos  = mp.Vector3(center_r, center_phi, center_z) 
    print(f"Source size: {source_size}")
    print(f"Source pos: {source_pos}")
elif coordinates=="CARTESIAN":
    cell_x = sx+2*dpml
    cell_y = sy+2*dpml
    cell_size = mp.Vector3(cell_x, cell_y, cell_z)
    cell_center_xy, cell_center_z = 0, 0  
    cell_center = mp.Vector3(cell_center_xy, cell_center_xy, cell_center_z) 
    dimensions = 3
    # SOURCE PROPERTIES
    center_x = 0.  # cell_x/2.
    center_y = 0.  # cell_y/2.
    center_z = -sz/2.
    source_size = mp.Vector3(cell_x, cell_y,  0.0) 
    source_pos  = mp.Vector3(center_x, center_y, center_z) 
    print(f"Source size: {source_size}")
    print(f"Source pos: {source_pos}")

# PML setup
pml_layers = [mp.PML(thickness=dpml)] 


# def parabola_shape(p):    #p... pixel
#     r = (p.x**2+p.y**2)**0.5
#     parabola_r = 2*cmath.sqrt(focal_len*(base_point-p.z-len_parabola_metal/2.))
#     if (p.z>(-len_parabola_metal)/2) and (p.z<(len_parabola_metal/2)+delta_parabola_metal) and (r>parabola_r.real):
#         return Au_JC_visible#mp.Medium(epsilon=55)#materials.Au_JC_visible
#     else:
#         return mp.vacuum#vacuum  i


# Final simulation box geometry
geometry = [mp.Block(
            size=cell_size,
            center=cell_center,  # Offset from simulation center
            # material=lambda p: parabola_shape(p, len_metal, focal_len, base_point, delta_metal, d_particle, mp.vacuum, Au_JC_visible, Y2O3, fused_quartz, offset_part))
            material=mp.vacuum)
            ]

# Time step setup
courant_factor = const.c*temp_stepsize/(1E-6/resolution)
readout_timestep = readout_fac * courant_factor / resolution
print(f"Courant factor (MEEP UNITS): {courant_factor}")
print(f"Readout timestep (MEEP UNITS): {readout_timestep}")





# ===================== Define laser =====================


def cart_cyl(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def W(z, z_offset, wlen, waist):
    Wm=waist
    ZR=np.pi*Wm**2/wlen
    return Wm*np.sqrt(1+np.power((z-z_offset)/ZR, 2))  


def R_inv(z, z_offset, wlen, waist):
    Wm=waist
    ZR=np.pi*Wm**2/wlen
    return (z-z_offset)/((z-z_offset)**2+ZR**2)


def LG_cyl_symm(rho, rho_offset, z, z_offset, wlen, waist, j_amp):
    k=2*np.pi/wlen
    Wm=waist
    Wz=W(z, z_offset, wlen, waist)
    r_inv=R_inv(z, z_offset, wlen, waist)
    ZR=np.pi*Wm**2/wlen
    return j_amp*np.sqrt(2/np.pi)*(1/Wz)*(np.sqrt(2)*(rho-rho_offset)/Wz)*\
        np.exp(-(rho-rho_offset)**2 / Wz**2)*\
        np.exp(-2*1j*np.arctan((z-z_offset)/ ZR))*\
        np.exp(0.5*1j*k*r_inv*(rho-rho_offset)**2)


def LG_cart_symm(x, x_offset, y, y_offset, z, z_offset, wlen, waist, j_amp, pol):
    k=2*np.pi/wlen
    rho, phi = cart_cyl(x-x_offset, y-y_offset)
    Wm=waist
    Wz=W(z, z_offset, wlen, waist)
    r_inv=R_inv(z, z_offset, wlen, waist)
    ZR=np.pi*Wm**2/wlen
    pre_fac = j_amp*np.sqrt(2/np.pi)*(1/Wz)*(np.sqrt(2)*(rho)/Wz)*\
    np.exp(-(rho)**2 / Wz**2)*\
    np.exp(-2*1j*np.arctan((z-z_offset)/ ZR))*\
    np.exp(0.5*1j*k*r_inv*(rho)**2)    
    if pol=="x":
        return -1/(2j)*pre_fac*(np.exp(1j*phi)-np.exp(-1j*phi))
    if pol=="y":
        return 1/2*pre_fac*(np.exp(1j*phi)+np.exp(-1j*phi))
    if pol=="rad_x":
        return 1/2*pre_fac*(np.exp(1j*phi)+np.exp(-1j*phi))
    if pol=="rad_y":
        return -1/(2j)*pre_fac*(np.exp(1j*phi)-np.exp(-1j*phi))


def sin2_func(t):    # , t_pulse_start, t_pulse_rise, t_pulse_flat, t_pulse_fall):
    temp_prop = np.exp(-1j * 2 * np.pi * fcen * (t - t_pulse_start))  
    if t < t_pulse_start:                                                                                                             
        return 0.j                                                                                                                    
    elif t < (t_pulse_start + t_pulse_rise):                                                                                          
        return 1j * np.sin(np.pi / 2 * (t - t_pulse_start) / t_pulse_rise)**2 * temp_prop 
    elif t < (t_pulse_start + t_pulse_rise + t_pulse_flat):                                                                           
        return 1j * temp_prop
    elif t < (t_pulse_start + t_pulse_rise + t_pulse_flat + t_pulse_fall):                                                            
        return 1.j * np.sin(np.pi / 2 * (t_pulse_start + t_pulse_rise + t_pulse_flat + t_pulse_fall - t) / t_pulse_fall               
                            )**2 * temp_prop 
    else:                                                                                                                             
        return 0.j

if True:
    time_source = mp.CustomSource(                   
        src_func= lambda t: sin2_func(t),                                                                                                                                         
        start_time=t_pulse_start,          
        end_time=runtime,                  
        center_frequency=fcen,             
        is_integrated=True                 
        )
else:
    time_source = mp.ContinuousSource(frequency=fcen, is_integrated=True),


if coordinates=="CYLINDRICAL":
    Source_Azi_LG =   [
        mp.Source(          
            time_source,
            center = source_pos,
            size = source_size, 
            component = mp.Ep,  
            amp_func = lambda r: LG_cyl_symm(r.x, -center_r, r.z, -center_z, wvl, w0, j_amp)
        )
         ]
    Source_Rad_LG =   [
        mp.Source(          
            time_source,
            center = source_pos,
            size = source_size, 
            component = mp.Er,  
            amp_func = lambda r: LG_cyl_symm(r.x, -center_r, r.z, -center_z, wvl, w0, j_amp)
        )
         ]

elif coordinates=="CARTESIAN":
    Source_Azi_LG =   [
        mp.Source(          
            time_source,                                                                                                                     
            center = source_pos,
            size = source_size, 
            component = mp.Ex,  
            amp_func = lambda r: LG_cart_symm(r.x, -center_x, r.y, -center_y, r.z, -center_z, wvl, w0, j_amp, "x")
        ),
        mp.Source(
            time_source,                                                                                                                   
            center = source_pos,
            size = source_size, 
            component = mp.Ey,  
            amp_func = lambda r: LG_cart_symm(r.x, -center_x, r.y, -center_y, r.z, -center_z, wvl, w0, j_amp, "y")
        ) 
         ]
    Source_Rad_LG =   [
        mp.Source(          
            time_source,                                                                                                                     
            center = source_pos,
            size = source_size, 
            component = mp.Ex,
            amp_func = lambda r: LG_cart_symm(r.x, -center_x, r.y, -center_y, r.z, -center_z, wvl, w0, j_amp, "rad_x")
        ),
        mp.Source(
            time_source,                                                                                                                   
            center = source_pos,
            size = source_size, 
            component = mp.Ey,  
            amp_func = lambda r: LG_cart_symm(r.x, -center_x, r.y, -center_y, r.z, -center_z, wvl, w0, j_amp, "rad_y")
        ) 
         ]

if beam == "azi":
    Source_LG = Source_Azi_LG
elif beam == "rad":
    Source_LG = Source_Rad_LG






# ===================== Setup simulation =====================

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    sources=Source_LG,
                    geometry=geometry,
                    geometry_center=mp.Vector3(-1/(2*resolution),-1/(2*resolution),-1/(2*resolution)),
                    Courant = courant_factor, 
                    dimensions=dimensions,
                    force_complex_fields = False,
                    default_material = mp.vacuum,
                    boundary_layers = pml_layers)


# ===================== Collect and print meta information =====================

if coordinates=="CYLINDRICAL":
    sim_center, sim_size = mp.visualization.get_2D_dimensions(sim, None)
    print("SIMULATION GEOMETRY")
    print("Desired simulation volume (cell_r, cell_phi, cell_z): ", (cell_r, cell_phi, cell_z))
    print("Simulation volume (sim.cell_size): ", sim.cell_size)
    print("Simulation center (sim.geometry_center): ", sim.geometry_center)
    print("Simulation center (sim_center): ", sim_center)
    print("Simulation size (sim_size): ", sim_size)
    print('-----------------------\n')
    xmin, xmax, ymin, ymax, zmin, zmax = mp.visualization.box_vertices(sim_center, sim_size, is_cylindrical = sim.is_cylindrical)
    Nrx, Nz = [int(np.linalg.norm(xmax - xmin)*resolution), int(np.linalg.norm(zmax - zmin)*resolution)]
elif coordinates=="CARTESIAN":
    #sim_center, sim_size = mp.visualization.get_2D_dimensions(sim, mp.Volume(center=sim.geometry_center, size=mp.Vector3(cell_x, 0, cell_z)))
    # sim.geometry_center, sim.cell_size = mp.visualization.get_2D_dimensions(sim, mp.Volume(center=sim.geometry_center, size=mp.Vector3(cell_x, cell_y, cell_z))) 
    print("SIMULATION GEOMETRY")
    print("Desired simulation volume (cell_xy, cell_z): ", (cell_x, cell_y, cell_z))
    print("Simulation volume (sim.cell_size): ", sim.cell_size)
    print("Simulation center (sim.geometry_center): ", sim.geometry_center)
    #print("Simulation center (sim_center): ", sim_center)
    #print("Simulation size (sim_size): ", sim_size)
    print('-----------------------\n')
    xmin, xmax, ymin, ymax, zmin, zmax = mp.visualization.box_vertices(sim.geometry_center, sim.cell_size, is_cylindrical = sim.is_cylindrical)
    Nrx, Ny, Nz = [int(np.linalg.norm(xmax - xmin)*resolution), int(np.linalg.norm(ymax - ymin)*resolution), int(np.linalg.norm(zmax - zmin)*resolution)]
# in cylindrical coordinates, radial (R) axis
# is in the range (0, R) rather than (-R/2, +R/2)
grid_resolution = sim.resolution



Nt_sim = int(runtime/(courant_factor/resolution)+1) if (runtime/(courant_factor/resolution))%1!=0 else int(runtime/(courant_factor/resolution))
print("Simulation output geometry")
print(f"(xmin, xmax, #sim, #saved): {xmin:.2f}, {xmax:.2f}, {Nrx:.2f}, {Nrx:.2f}")
print(f"(ymin, ymax, #sim, #saved): {ymin:.2f}, {ymax:.2f}, {Ny:.2f}, {Ny:.2f}")
print(f"(zmin, zmax, #sim, #saved): {zmin:.2f}, {zmax:.2f}, {Nz:.2f}, {Nz:.2f}")
print(f"(tmin, tmax, #sim, #saved): {0:.2f}, {runtime:.2f}, {Nt_sim:.2f}")
print('-----------------------\n')
print("Internal resolution in s: %.2e" % (courant_factor / grid_resolution / const.c * 1E-6) )

if coordinates=="CYLINDRICAL":
    r_arr = np.linspace(xmin, xmax, Nrx)*1E-6  # starting from 0
    r_arr_wo_dpml_idx = bisect.bisect_left(r_arr, (xmax-dpml)*1E-6)
    rx_arr = np.linspace(-xmax, xmax, 2*Nrx-1)*1E-6
    # rx_arr_wo_dpml_idx = [bisect.bisect_right(rx_arr, (-xmax-dpml)*1E-6), bisect.bisect_left(rx_arr, (xmax-dpml)*1E-6)]
    # rx_arr_wo_dpml = np.linspace(-xmax+dpml, xmax-dpml, 2*Nrx-1)*1E-6 
    memory_space = Nt_sim*Nrx*Nz*16
    disk_space = Nt_sim/readout_fac*Nrx*Nz*8*6  # 16bytes for complex number, 6 for 6 exported fields (3comp of B and 3 comp of E)

elif coordinates=="CARTESIAN":
    rx_arr = np.linspace(xmin, xmax, Nrx)*1E-6
    # rx_arr_wo_dpml_idx = [bisect.bisect_right(rx_arr, (xmin+dpml)*1E-6), bisect.bisect_left(rx_arr, (xmax-dpml)*1E-6)]
    # rx_arr_wo_dpml = np.linspace(xmin+dpml, xmax-dpml, 2*Nrx-1)*1E-6 
    y_arr = np.linspace(ymin, ymax, Ny)*1E-6
    # y_arr_wo_dpml_idx = [bisect.bisect_right(y_arr, (ymin+dpml)*1E-6), bisect.bisect_left(y_arr, (ymax-dpml)*1E-6)]
    # y_arr_wo_dpml = np.linspace(ymin+dpml, ymax-dpml, 2*Ny-1)*1E-6 
    memory_space = Nt_sim*Nrx*Ny*Nz*16
    disk_space = Nt_sim/readout_fac*Nrx*Ny*Nz*8*6  # 16bytes for complex number, 6 for 6 exported fields (3comp of B and 3 comp of E)

z_arr = np.linspace(zmin, zmax, Nz)*1E-6 
# z_arr_wo_dpml_idx = [bisect.bisect_right(z_arr, (zmin+dpml)*1E-6), bisect.bisect_left(z_arr, (zmax-dpml)*1E-6)] 
# z_arr_wo_dpml = z_arr[z_arr_wo_dpml_idx[0]: z_arr_wo_dpml_idx[1]]
# z_arr_metal = np.linspace(-len_metal/2., len_metal/2.+delta_metal, Nz)*1E-6





# ===================== Save results =====================

def save_hdf5():
    sim_data = h5py.File("sim_file.hdf5", "w")
    sim_data.create_dataset("e_x_data_si", data=ex_long_list)
    sim_data.create_dataset("e_y_data_si", data=ey_long_list)
    sim_data.create_dataset("e_z_data_si", data=ez_long_list)
    sim_data.create_dataset("b_x_data_si", data=bx_long_list)
    sim_data.create_dataset("b_y_data_si", data=by_long_list)
    sim_data.create_dataset("b_z_data_si", data=bz_long_list) 
    sim_data.attrs["tmax_si"] = runtime /const.c *1E-6 
    sim_data.attrs["sim_dt_si"] = temp_stepsize
    sim_data.attrs["saved_dt_si"] = readout_timestep /const.c * 1E-6
    sim_data.attrs["dpml_rx_si"] = dpml*1E-6
    sim_data.attrs["rxmin_si_output"] = rx_output_arr[0][0]*1E-6
    sim_data.attrs["rxmax_si_output"] = rx_output_arr[0][1]*1E-6
    sim_data.attrs["drx_si_output"] = rx_output_arr[0][2]*1E-6 
    if coordinates=="CARTESIAN": 
        sim_data.attrs["dpml_y_si"] = dpml*1E-6 
        sim_data.attrs["ymin_si_output"] = y_output_arr[0][0]*1E-6
        sim_data.attrs["ymax_si_output"] = y_output_arr[0][1]*1E-6
        sim_data.attrs["dy_si_output"] = y_output_arr[0][2]*1E-6
    sim_data.attrs["dpml_z_si"] = dpml*1E-6  
    sim_data.attrs["zmin_si_output"] = z_output_arr[0][0]*1E-6
    sim_data.attrs["zmax_si_output"] = z_output_arr[0][1]*1E-6
    sim_data.attrs["dz_si_output"] = z_output_arr[0][2]*1E-6 
    sim_data.attrs["dimensions"] = coordinates
    sim_data.close()


print(f"Min. expected memory consumption: {memory_space/1E9:.1E} GB")
print(f"Min. expected disk space consumption: {disk_space/1E9:.1E} GB")


bx_long_list = []                                                                                                                         
by_long_list = []                                                                                                                         
bz_long_list = []                                                                                                                         
ex_long_list = []                                                                                                                         
ey_long_list = []                                                                                                                         
ez_long_list = []  
rx_output_arr = []  
y_output_arr =  []
z_output_arr =  []


def get_slice_long(sim):                                                                                                                  
    if coordinates=="CYLINDRICAL":
        bx_long_list.append(sim.get_array(center=sim_center, size=sim_size, component=mp.Br)*b_conv_fac)
        by_long_list.append(sim.get_array(center=sim_center, size=sim_size, component=mp.Bp)*b_conv_fac)
        bz_long_list.append(sim.get_array(center=sim_center, size=sim_size, component=mp.Bz)*b_conv_fac)
        ex_long_list.append(sim.get_array(center=sim_center, size=sim_size, component=mp.Er)*e_conv_fac)
        ey_long_list.append(sim.get_array(center=sim_center, size=sim_size, component=mp.Ep)*e_conv_fac)
        ez_long_list.append(sim.get_array(center=sim_center, size=sim_size, component=mp.Ez)*e_conv_fac)
    elif coordinates=="CARTESIAN":
        bx_long_list.append(sim.get_array(center=sim.geometry_center, size=sim.cell_size, component=mp.Bx)*b_conv_fac)
        by_long_list.append(sim.get_array(center=sim.geometry_center, size=sim.cell_size, component=mp.By)*b_conv_fac)
        bz_long_list.append(sim.get_array(center=sim.geometry_center, size=sim.cell_size, component=mp.Bz)*b_conv_fac)
        ex_long_list.append(sim.get_array(center=sim.geometry_center, size=sim.cell_size, component=mp.Ex)*e_conv_fac)
        ey_long_list.append(sim.get_array(center=sim.geometry_center, size=sim.cell_size, component=mp.Ey)*e_conv_fac)
        ez_long_list.append(sim.get_array(center=sim.geometry_center, size=sim.cell_size, component=mp.Ez)*e_conv_fac)


def get_coordinates(sim):
    x, y, z, w =sim.get_array_metadata(vol=mp.Volume(center=sim.geometry_center, size=sim.cell_size))
    print(f"x_arr {x}")
    print(f"xmin, xmax, dx, len: {x[0], x[-1], x[1]-x[0], len(x)}")
    ex_slice = sim.get_array(center=sim.geometry_center, size=sim.cell_size, component=mp.Ex)
    print(f"ex_slice.shape: {ex_slice.shape}")
    print(f"ex_slice {ex_slice[:, len(ex_slice[1])//2, len(ex_slice[2])//20]}")
    print(f"z_arr {z}")
    print(f"zmin, zmax, dz, len: {z[0], z[-1], z[1]-z[0], len(z)}")
    print(sim.geometry_center.x, sim.geometry_center.y, sim.geometry_center.z)
    print(sim.cell_size.x, sim.cell_size.y, sim.cell_size.z)
    rx_output_arr.append([x[0], x[-1], x[1]-x[0]])
    rx_output_arr.append([x[0], x[-1], x[1]-x[0]])
    rx_output_arr.append([x[0], x[-1], x[1]-x[0]])
    y_output_arr.append([y[0], y[-1], y[1]-y[0]])
    z_output_arr.append([z[0], z[-1], z[1]-z[0]])


sim.run(mp.at_beginning(get_slice_long),                                                                                                      
        mp.at_end(get_coordinates),
        mp.at_every(readout_timestep, get_slice_long),  
        until=runtime
        )
save_hdf5()

