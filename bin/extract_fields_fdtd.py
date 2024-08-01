#!/usr/bin/env python3                                                                                                                                  

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2023 University of Vienna
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

# Interactive script for the extraction of EM-fields from FDTD simulations to a laser input file for SHARC
#
# usage: python extract_fields_fdtd.py


import numpy as np 
import h5py
import os
import datetime
import time
import sys
import scipy.constants as const  # SHOULD THIS BE WRITTEN in the constants library? 
from scipy.signal import find_peaks
import shutil
import matplotlib.pyplot as plt
import pywt

from logger import log
from scipy import fft, signal, ndimage
from utils import question                                 
from scipy.interpolate import RegularGridInterpolator
# from SHARC_INTERFACE import SHARC_INTERFACE                
# =========================================================
sharcversion='4.0'  # QA -> Take from SHARC

version = '1.0'                                                                                                                                
versionneeded = [1.0, float(version)]                                                                                           
versiondate = datetime.date(2023, 8, 24)                                                                                                       
global KEYSTROKES                                                                                                                              
old_question = question

# UNIT FACTORS
spat_unit_fac = 1E-6  # Conversion input unit to SI
temp_unit_fac = 1E-15  # Conversion input unit to SI
# stepsize = 0.5  # Length of the nuclear dynamics time steps in fs: QA -> take from SHARC
# nsubsteps = 25  # Number of substeps for the integration of the electronic EOM: QA -> take from SHARC
efield_au_to_v_per_m = const.physical_constants["Hartree energy"][0]/const.e/const.physical_constants["Bohr radius"][0]
bfield_au_to_t = const.electron_mass*const.physical_constants["Hartree energy"][0]/(const.e*const.physical_constants["reduced Planck constant"][0])
efield_grad_au_to_v_per_m2 = efield_au_to_v_per_m/const.physical_constants["Bohr radius"][0]     
bfield_grad_au_to_t_per_m =  bfield_au_to_t/const.physical_constants["Bohr radius"][0]    


int_method = "cubic"                 
space_tolerance = 2  # only one works for now
time_tolerance = 2  # only one works for now

progress_width = 50
write_shift=0
sim_file_attrs = ["dimensions", "tmax_si", "rxmin_si_output", "rxmax_si_output", "ymin_si_output", "ymax_si_output", "zmin_si_output", "zmax_si_output"]

efields = ["e_x_data_si", "e_y_data_si", "e_z_data_si"]
bfields = ["b_x_data_si", "b_y_data_si", "b_z_data_si"]

def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    return old_question(question=question, typefunc=typefunc, KEYSTROKES=KEYSTROKES, default=default, autocomplete=autocomplete, ranges=ranges)


def try_read(word, index, typefunc, default):                                                                           
    try:                                                                                                                
        return typefunc(word[index])                                                                                    
    except IndexError:                                                                                                  
        return typefunc(default)                                                                                        
    except ValueError:                                                                                                  
        log.info('Could not initialize object!')                                                                        
        quit(1)                                                                                                         



def custom_formatter(val: float):
    """
    Formats the laser fields files' values in defined scientific notation
    Args:
        x (int): 

    Returns:
       Formatted laser fields files' values 
    """
    assert isinstance(val, float), "val must be a float!"
    if val!=0.0:
        if np.abs(val)<1E-99:
            val=0.0
    val_form = '{:.8e}'.format(val)  # Format with 3 digits for the exponent
    mantissa, exponent = val_form.split('e')
    sign = '  ' if float(mantissa) >= 0 else ' '  # Check if positive
    return f'{sign}{mantissa}E{exponent[0]}{exponent[1:].zfill(2)}'


def displaywelcome():
    log.info('Script for extraction of laser fields from FDTD simulation output and creation of a laser field file started...\n')
    string = '\n'
    string += '  ' + '=' * 80 + '\n'
    input = [' ',
             'Setup laser fields file for SHARC dynamics',
             ' ',
             'Authors: Lorenz Grünewald',
             ' ',
             'Version: %s' % (version),
             'Date: %s' % (versiondate.strftime("%d.%m.%y")),
             ' ']
    for inp in input:
        string += '||{:^80}||\n'.format(inp)
    string += '  ' + '=' * 80 + '\n\n'
    string += '''
This script automatizes the extraction of laser fields from a MEEP FDTD simulation at one spatial point and creates a laser field file
for SHARC dynamics.
  '''
    log.info(string)


def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open('KEYSTROKES.temp', 'w')


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move('KEYSTROKES.temp', 'KEYSTROKES.extract_laser_fields')


def get_general(INFOS):
    '''This routine questions from the user some general information:
    - FDTD simulation output file
    - temporal stepsize (for interpolation)
    - spatial (3D) stepsize (for interpolation)
    - spatial (3D) point at which the fields should be extracted'''

    log.info(f'{"FDTD simulation file":-^60s}' + '\n')
    # open the initconds file
    try:
        sim_file_path = 'sim_file.hdf5'
        if os.path.exists(sim_file_path):
            log.info('FDTD simulation output file "sim_file.hdf5" detected. Do you want to use this?')
            if not question('Use file "sim_file.hdf5"?', bool, True):
                raise IOError
        else:
            raise IOError
    except IOError:
        log.info('\nIf you do not have an FDTD output file, prepare one with MEEP!\n')
        log.info('Please enter the filename of the FDTD simulation output file.')
        while True:
            sim_file_path = question('FDTD simulation output filename:', str, 'FDTD output')
            sim_file_path = os.path.expanduser(os.path.expandvars(sim_file_path))
            if os.path.isdir(sim_file_path):
                log.info(f'Is a directory: {sim_file_path}')
                continue
            if not os.path.isfile(sim_file_path):
                log.info(f'File does not exist: {sim_file_path}')
                continue
            else:
                break
    try:
        sim_file = h5py.File(sim_file_path, 'r')
    except IOError:
        log.info('Could not open: {sim_file_path}')
    for attrs in sim_file_attrs:
        try:
            sim_file.attrs[attrs]
            pass
        except KeyError:
            log.info(f'Could not find attribute "{attrs}" in provided simulation output file: {sim_file_path}')
            log.info(f'Complete list of attributes: {list(sim_file.attrs)}')
            raise KeyError
    INFOS["sim_file_path"] = sim_file_path
    INFOS["dimensions"] = sim_file.attrs["dimensions"]
    if sim_file.attrs["dimensions"]=="CARTESIAN":
        INFOS["tmin"], INFOS["xmin"], INFOS["ymin"], INFOS["zmin"] = [0] + [sim_file.attrs[var] for var in ["rxmin_si_output", "ymin_si_output", "zmin_si_output"]]
        INFOS["tmax"], INFOS["xmax"], INFOS["ymax"], INFOS["zmax"] = [sim_file.attrs[var] for var in ["tmax_si", "rxmax_si_output", "ymax_si_output", "zmax_si_output"]] 
        # tmin, tmax, tres = [0, sim_file.attrs["tmax_si"], readout_file.attrs["saved_dt_si"]]
        # rxres, zres = [sim_file.attrs["drx_si_output"], readout_file.attrs["dz_si_output"]]
        INFOS["Nt"], INFOS["Nx"], INFOS["Ny"], INFOS["Nz"] = sim_file["e_x_data_si"].shape
    else:
        # INFOS['tmin'], INFOS['xmin'], INFOS['zmin'] = [0] + [sim_file.attrs[var] for var in ["rxmin_si_output", "zmin_si_output"]]
        # INFOS['tmax'], INFOS['xmax'], INFOS['zmax'] = [sim_file.attrs[var] for var in ["tmax_si", "rxmax_si_output", "zmax_si_output"]] 
        # INFOS['Nt'], INFOS['Nrx'], INFOS['Nz'] = sim_file['e_x_data_si'].shape
        log.info('Cylindrical coordinates not implemented yet!')
        raise IOError
    log.info(f'\nFile "{sim_file_path}" contains simulation output in {sim_file.attrs["dimensions"]} coordinates.')
    log.info("Fields are saved within the following coordinates:")
    
    if sim_file.attrs['dimensions']=="CARTESIAN":
        log.info(f'x (µm): ({INFOS["xmin"]/spat_unit_fac:.2f}, {INFOS["xmax"]*1E6:.2f})') 
        log.info(f'y (µm): ({INFOS["ymin"]/spat_unit_fac:.2f}, {INFOS["ymax"]/spat_unit_fac:.2f})')
    log.info(f'z (µm): ({INFOS["zmin"]/spat_unit_fac:.2f}, {INFOS["zmax"]/spat_unit_fac:.2f})')
    log.info("------------------------------")
    # QA: logging module probably has some problems with f-strings 
    #   -> Should I switch to other formatting?
    # QA: Which units should be default in input?
    #   -> Would suggest µm, Angstrom
    x_mean, y_mean, z_mean = [(INFOS["xmax"]/spat_unit_fac+INFOS["xmin"]/spat_unit_fac)/2, 
                              (INFOS["ymax"]/spat_unit_fac+INFOS["ymin"]/spat_unit_fac)/2,
                              (INFOS["zmax"]/spat_unit_fac+INFOS["zmin"]/spat_unit_fac)/2]
    log.info(f'\nPlease enter the laser field extraction positions (in µm) as three floats separated by space. Default: [{x_mean:.2f}, {y_mean:.2f}, {z_mean:.2f}]')
    while True:
        extract_point = question('Extraction point:', float, [x_mean, y_mean, z_mean])  # Default extraction point at equilibrium
        if len(extract_point) != 3:
            log.info('Enter three numbers separated by spaces!')
            continue
        if (INFOS["xmin"] > extract_point[0]*spat_unit_fac) or (INFOS["xmax"] < extract_point[0]*spat_unit_fac):
            log.info(f'X-coordinate of extraction point {extract_point[0]} must lie within ({INFOS["xmin"]/spat_unit_fac:.2f}, {INFOS["xmax"]/spat_unit_fac:.2f}) \u03bcm !')
            continue
        if (INFOS["ymin"] > extract_point[1]*spat_unit_fac) or (INFOS["ymax"] < extract_point[1]*spat_unit_fac):
            log.info(f'Y-coordinate of extraction point {extract_point[1]} must lie within ({INFOS["ymin"]/spat_unit_fac:.2f}, {INFOS["ymax"]/spat_unit_fac:.2f}) \u03bcm !')
            continue
        if (INFOS["zmin"] > extract_point[2]*spat_unit_fac) or (INFOS["zmax"] < extract_point[2]*spat_unit_fac):
            log.info(f'Z-coordinate of extraction point {extract_point[2]} must lie within ({INFOS["zmin"]/spat_unit_fac:.2f}, {INFOS["zmax"]/spat_unit_fac:.2f}) \u03bcm !')
            continue
        break
    extract_point_fmt = ', '.join([f"{val:.2f}" for val in extract_point])
    log.info(f'Script will extract fields at [{extract_point_fmt}] \u03bcm.')
    INFOS["extract_point"] = [coord*spat_unit_fac for coord in extract_point]
    # QA: time step / resolution
    #   -> what should be the default, default unit
    log.info('\nPlease enter the desired number of electronic time steps within a nuclear dynamics time step [Must match with SHARC nsubsteps]. Default: 25')
    while True:
        no_el_time_step = question('Number of time steps:', float, [25])  # Default time step 
        if len(no_el_time_step) != 1:
            log.info('Enter one time step!')
            continue
        break
    INFOS["nuc_dyn_stepsize"] = INFOS["tmax"]/(INFOS["Nt"]-1)
    INFOS["electronic time_step"] = INFOS["nuc_dyn_stepsize"]/no_el_time_step[0] 
    while True:
        log.info('\nPlease enter the desired spatial interpolation step (in nm). Default: 10')
        delta = question('dx/dy/dz:', float, [10])
        if len(no_el_time_step) != 1:
            log.info('Enter one time step!')
            continue
        break
    INFOS["delta"] = delta[0]*1E-9
    log.info('\nWhich fields/gradients do you want to export? Default: y')
    export_e = question('Export electric field:', bool, True)  # Default time step 
    export_b = question('Export magnetic field:', bool, True)  # Default time step 
    export_egrad = question('Export electric field gradients:', bool, True)  # Default time step 
    export_bgrad = question('Export magnetic field gradients:', bool, True)  # Default time step 
    if not export_e and not export_b and not export_egrad and not export_bgrad: 
        log.info('Nothing to export!')
        raise IOError
        # QA: Does one have to return a value, if raise IOError?
    INFOS["export_e"]=export_e
    INFOS["export_b"]=export_b
    INFOS["export_egrad"]=export_egrad
    INFOS["export_bgrad"]=export_bgrad
    return INFOS


# def calc_fields(t_i, point, point_idx, delta, quant, cmplx, method, tol, dim):
def calc_fields(INFOS, time_arr, rx_arr, y_arr, z_arr, quant_name: str, quant: np.ndarray, cmplx: str, readout_time: float, point_idx: list, space_tol: int, time_tol: int, int_method: str):
    assert isinstance(readout_time, float), "readout_time must be a float!"
    assert isinstance(point_idx, list), "point_ipoint_idx must be a list!" 
    assert isinstance(space_tol, int), "space_tol must be an integer!"
    assert isinstance(time_tol, int), "time_tol must be an integer!"
    assert isinstance(quant, np.ndarray), "quant must be an array!"
    assert isinstance(quant_name, str), "quant name must be a string!"
    # QA: Should I couple the tolerance directly to the tolerance or give an error if cubic is expected and tol=1?
    assert isinstance(int_method, str), "int_method must be a string!"
    assert isinstance(cmplx, str), "cmplx must be a string!"
    

    if INFOS["dimensions"]=="CARTESIAN": 
        point_idt = np.argmin(np.abs(readout_time-time_arr))
        grid_idx_x = (point_idx[0]-space_tol, point_idx[0]+space_tol+1)
        grid_idx_y = (point_idx[1]-space_tol, point_idx[1]+space_tol+1)
        grid_idx_z = (point_idx[2]-space_tol, point_idx[2]+space_tol+1)
        dx_basis, dy_basis, dz_basis = [np.eye(3)[idx, :]*INFOS["delta"] for idx in range(3)]

        if (point_idt-time_tol)<=0 or (point_idt+time_tol)>=len(time_arr):  # Time point at border of simulated time 
            grid = (rx_arr[grid_idx_x[0]:grid_idx_x[1]],
                    y_arr[grid_idx_y[0]:grid_idx_y[1]],
                    z_arr[grid_idx_z[0]:grid_idx_z[1]])
            interpol_point = INFOS["extract_point"]
            if cmplx=="real":
                interp = RegularGridInterpolator(grid, np.real(quant[
                                point_idt,
                                point_idx[0]-space_tol:point_idx[0]+space_tol+1,  # No fancy indexing allowed!                                                                                                                                                                                                                              
                                point_idx[1]-space_tol:point_idx[1]+space_tol+1,                                                                                                                                                                                                                              
                                point_idx[2]-space_tol:point_idx[2]+space_tol+1]),
                                                 method=int_method)
            elif cmplx=="imag":
                interp = RegularGridInterpolator(grid, np.imag(quant[
                                point_idt, 
                                point_idx[0]-space_tol:point_idx[0]+space_tol+1,
                                point_idx[1]-space_tol:point_idx[1]+space_tol+1,
                                point_idx[2]-space_tol:point_idx[2]+space_tol+1]),
                                                 method=int_method)
        else:  # Also interpolate in time
            red_time_arr_idx = (point_idt-time_tol, point_idt+time_tol+1) 
            red_time_arr = time_arr[red_time_arr_idx[0]:red_time_arr_idx[1]]
            grid = (red_time_arr,
                    rx_arr[grid_idx_x[0]:grid_idx_x[1]],
                    y_arr[grid_idx_y[0]:grid_idx_y[1]],
                    z_arr[grid_idx_z[0]:grid_idx_z[1]]) 
            interpol_point=[readout_time, *INFOS["extract_point"]]
            dx_basis , dy_basis, dz_basis = np.array([[0, *dx_basis], [0, *dy_basis], [0, *dz_basis]])
            if cmplx=="real":
                interp = RegularGridInterpolator(grid, np.real(quant[
                                     point_idt-time_tol:point_idt+time_tol+1,
                                     point_idx[0]-space_tol:point_idx[0]+space_tol+1,  # No fancy indexing allowed!                                                                                                                                                                                                                              
                                     point_idx[1]-space_tol:point_idx[1]+space_tol+1,                                                                                                                                                                                                                              
                                     point_idx[2]-space_tol:point_idx[2]+space_tol+1]),
                                                 method=int_method)
            elif cmplx=="imag":
                interp = RegularGridInterpolator(grid, np.imag(quant[
                                     point_idt-time_tol:point_idt+time_tol+1, 
                                     point_idx[0]-space_tol:point_idx[0]+space_tol+1,
                                     point_idx[1]-space_tol:point_idx[1]+space_tol+1,
                                     point_idx[2]-space_tol:point_idx[2]+space_tol+1]),
                                                 method=int_method) 
        fields = interp(interpol_point)[0]
        if ((quant_name in efields) and INFOS["export_egrad"]) or ((quant_name in bfields) and INFOS["export_bgrad"]):
            gradients = [(interp((interpol_point+di_basis))[0]- interp((interpol_point-di_basis))[0])/(2*INFOS["delta"]) for di_basis in [dx_basis, dy_basis, dz_basis]]
            
            return fields, *gradients 
        else:
            return [fields]
    else:
        log.info(f'Dimension not implemented yet: {INFOS["dimensions"]}')
        raise IOError
        return 0


def fft_calc(field, time_arr, zero_padding_factor=2):                                                              
    N_padded = len(time_arr)*zero_padding_factor
    field_padded = np.pad(field, (0, N_padded - len(time_arr)), 'constant')
    freq_signal = 2.0 / N_padded * np.abs(fft.fft(field_padded)[:N_padded//2])                                               
    freq = fft.fftfreq(N_padded, d=(time_arr[1]-time_arr[0]))[:N_padded//2]
    return freq, freq_signal


def wavelet_calc_sum(field, time_arr):
    dt = time_arr[1]-time_arr[0]                                                                   
    # freq_step = 1/dt                                                                               
    # freq_arr = np.linspace(1, freq_step/2, 100)
    f_min = const.c/1E-6  # IR light
    f_max = const.c/2E-7  # UV light 
    wavelet = 'cmor2-4'
    scales = np.linspace(*pywt.frequency2scale(wavelet, [f_min, f_max])/dt, num=int(1E3))  # lam_res = (lam_c-lam_max)/(const.c/d_nu), from max to min to get ascending frequencies in return
    #cwtmatr, freqs = pywt.cwt(field, scales=scales, wavelet="morl", sampling_period=dt)
    cwtmatr, freqs = pywt.cwt(field, scales=scales, wavelet=wavelet, sampling_period=dt)
    cwtmatr = np.abs(cwtmatr)
    freq_sig_integrated = np.sum(cwtmatr, axis=1)/len(time_arr)  # mean signal on frequencies over all times 
    return freqs, freq_sig_integrated                                                                                       


def wavelet_calc(field, time_arr):
    dt = time_arr[1]-time_arr[0]                                                                   
    # freq_step = 1/dt                                                                               
    # freq_arr = np.linspace(1, freq_step/2, 100)
    # f_min = 2.8E14  # IR light
    # f_max = 1E15  # UV light 
    # a_min = 1 / (f_max * dt)
    # a_max = 1 / (f_min * dt)
    # scales = np.linspace(a_max, a_min, num=int(1E3))  # lam_res = (lam_c-lam_max)/(const.c/d_nu)
    f_min = const.c/1E-6  # IR light
    f_max = const.c/2E-7  # UV light 
    wavelet = 'cmor2-4'
    scales = np.linspace(*pywt.frequency2scale(wavelet, [f_min, f_max])/dt, num=int(1E3))  # lam_res = (lam_c-lam_max)/(const.c/d_nu), from max to min to get ascending frequencies in return
    # cwtmatr, freqs = pywt.cwt(field, scales=scales, wavelet="morl", sampling_period=dt)
    cwtmatr, freqs = pywt.cwt(field, scales=scales, wavelet=wavelet, sampling_period=dt)
    # cwtmatr, freqs = pywt.cwt(field, scales=scales, wavelet="morl", sampling_period=dt)
    cwtmatr = np.abs(cwtmatr)
    # plt.plot(time_arr, field)
    # plt.show()
    return freqs, cwtmatr


def extract_frequencies(laser_file, avail_e, avail_b, avail_egrad, avail_bgrad, time_arr):
    while True:
        shift = 0
        em_fields = []
        if avail_e:
            efield = np.transpose(laser_file[:, 1:7:2] + 1.j * laser_file[:, 2:7:2])
            em_fields += [comp for comp in efield]
            shift+=6
        if avail_b:
            bfield = np.transpose(laser_file[:, 1 + shift:7 + shift:2] + 1.j * laser_file[:, 2 + shift:7 + shift:2])
            em_fields += [comp*const.c for comp in bfield]  # B-field amplitude match to E-field for weighting 
            shift+=6

        time_au_to_s = const.physical_constants["reduced Planck constant"][0]/const.physical_constants["Hartree energy"][0]
        while True:
            fft_field = question("Do you want to provide laser frequencies (0), perform FFT (1), perform WT (2)  or integrated WT (3)?", int, [0])
            match fft_field[0]:
                case 0:
                    i_unit = question("Frequency unit: (0) nm, (1) Hz, (2) eV, (3) a.u.", int, 0)
                    laser_frequencies = question("Provide frequency list:", list, [None])
                    match i_unit[0]:
                        case 0:
                            log.info(f"Provided frequencies: {laser_frequencies} in nm")
                            laser_frequencies = [time_au_to_s*(const.c/(freq*1E-9)) for freq in laser_frequencies]
                        case 1:
                            log.info(f"Provided frequencies: {laser_frequencies} in Hz")
                            laser_frequencies *= time_au_to_s
                        case 2: 
                            log.info(f"Provided frequencies: {laser_frequencies} in eV")
                            laser_frequencies *time_au_to_s/const.h 
                        case 3:
                            log.info(f"Provided frequencies: {laser_frequencies} in a.u.")
                        case _:
                            log.info(f"Did not understand input: {i_unit}!")
                case 1:
                    fft_freq = [fft_calc(em_fields[field_idx], time_arr)[0] for field_idx in range(len(em_fields))]
                    fft_signal = [fft_calc(em_fields[field_idx], time_arr)[1] for field_idx in range(len(em_fields))]
                    fft_signal_max = [(fft_calc(em_fields[field_idx], time_arr)[1]).max() for field_idx in range(len(em_fields))]
                    freq_peaks = [find_peaks(fft_signal[field_idx], prominence=fft_signal_max[field_idx]/3., height=1E-7) for field_idx in range(len(em_fields))]  # indices of found  peaks
                    # Peak criteria: 20% prominence relative to the max. height, bigger than twice the average and 1E-8
                    # for i in range(len(em_fields)):
                    #     plt.plot(fft_freq[i], fft_signal[i])
                    #     plt.plot(fft_freq[i][freq_peaks[i][0]], fft_signal[i][freq_peaks[i][0]], "rx")
                    #     plt.show()
                    #     # plt.plot(time_arr, em_fields[i], "b.")
                    #     # plt.show()
                    freq_ex = np.hstack([fft_freq[0][peak[0]] for peak in freq_peaks])
                    if len(freq_ex) == 0:
                        log.info('Found no distinct frequencies!')
                        raise IOError                                                    
                    return np.vstack([freq_ex for t_el in time_arr])
                    break
                case 2:
                    freqs, freq_signals_per_time = zip(*[wavelet_calc(em_fields[field_idx], time_arr) for field_idx in range(len(em_fields))])
                    freq_signals_summed_max = np.max(freq_signals_per_time, axis = (1, 2))  # Maximum over all times and frequencies for one field
                    max_no_of_peaks = 0 
                    for field_idx in range(len(em_fields)):
                        for t_idx in range(len(time_arr)):
                            a, _ = find_peaks(freq_signals_per_time[field_idx][:, t_idx], prominence=freq_signals_summed_max[field_idx]/5., height=1E-7)
                            if len(a) > max_no_of_peaks:
                                max_no_of_peaks = len(a)
                    freq_peaks = np.zeros((len(em_fields), len(time_arr), max_no_of_peaks), int)
                    for field_idx in range(len(em_fields)):
                        for t_idx in range(len(time_arr)):
                            freqs_field_time, _ = find_peaks(freq_signals_per_time[field_idx][:, t_idx], prominence=freq_signals_summed_max[field_idx]/5., height=1E-7)
                            freqs_field_time = freqs_field_time[freqs_field_time != len(freq_signals_per_time[0][:, 0])-1]  # filter out last value
                            if len(freqs_field_time) != 0:
                                freq_peaks[field_idx, t_idx, :len(freqs_field_time)] = np.array([freqs[0][freqs_field_time]])  # frequencies of peaks per field per time
                            #plot
                            #if len(freqs_field_time) != 0:
                            #    freq_peaks[field_idx, t_idx, :len(freqs_field_time)] = np.array([freqs_field_time])  # frequencies of peaks per field per time
                            # for t_i in range(len(time_arr)):
                            #     if freq_peaks[field_idx, t_i, 0] != int(0):
                            #         plt.plot(freqs[0], freq_signals_per_time[0][:, t_i], "b-")
                            #         for freq_idx in range(len(freq_peaks[0, 0, :])):
                            #             print(freq_peaks[field_idx, t_i, freq_idx])
                            #             plt.plot(freqs[0][freq_peaks[field_idx, t_i, freq_idx]], freq_signals_per_time[0][freq_peaks[field_idx, t_i, freq_idx], t_i], "rx")
                            #         plt.show()
                        #plt.pcolormesh(time_arr, freqs[field_idx], freq_signals_per_time[field_idx])#, aspect='auto', origin='lower', extent=[time_arr[0], time_arr[-1], freqs[field_idx][0], freqs[field_idx][-1]])
                        #plt.scatter(time_arr, freqs[0][freq_peaks[field_idx, :, :]])
                        #plt.show()
                    freq_peaks = np.einsum('ijk->jki', freq_peaks).reshape(len(time_arr), -1)
                    freq_peaks = freq_peaks[:, np.any(freq_peaks != 0, axis=0)] 
                    freq_peaks = np.unique(freq_peaks, axis=1)
                    return freq_peaks 
                    break
                case 3:
                    freqs, freq_signals_summed = zip(*[wavelet_calc_sum(em_fields[field_idx], time_arr) for field_idx in range(len(em_fields))])
                    freq_signals_summed_max = np.max(freq_signals_summed, axis = 1)
                    freq_peaks, _ = zip(*[find_peaks(freq_signals_summed[field_idx], prominence=freq_signals_summed_max[field_idx]/5., height=1E-7) for field_idx in range(len(em_fields))])
                    freq_peaks = np.hstack(freq_peaks)
                    if len(freq_peaks) != 0:
                        print(freq_peaks)
                        freq_peaks = np.array([freqs[0][freq] for freq in freq_peaks])
                    # for i in range(len(em_fields)):
                    #     plt.plot(freqs[i], freq_signals_summed[i])
                    #     plt.plot(freqs[i][freq_peaks[i][0]], freq_signals_summed[i][freq_peaks[i][0]], "rx")
                    #     plt.show()
                    return np.vstack([freq_peaks for t_idx, t_el in enumerate(time_arr)])
                    break
                case _:
                    log.info(f"Did not understand input: {fft_field}!")
                    continue


def main():
    '''Main routine'''

    usage = '''
    python extract_fields_fdtd.py 
    Interactive script for the extraction of EM-fields from FDTD simulations to a laser input file for SHARC
    As input it takes an FDTD output (.hdf5), the spatial position of the fields to be extracted and the time step to be interpolated
    '''

    # description = ''
    # parser = OptionParser(usage=usage, description=description)

    displaywelcome()
    open_keystrokes()
    INFOS = {}
    INFOS['cwd'] = os.getcwd()

    INFOS = get_general(INFOS)
    for item in INFOS:
        log.info(f"{item:<25} {INFOS[item]}")  
    extract = question("Do you want to perform the specified EM-Field extraction?", bool, True) 
    log.info("")                                                                     
    if extract:
        t_arr = np.linspace(INFOS["tmin"], INFOS["tmax"], INFOS["Nt"], endpoint=True)  
        int_t_arr = np.arange(INFOS["tmin"], INFOS["tmax"]+INFOS["electronic time_step"], INFOS["electronic time_step"])
        rx_arr = np.linspace(INFOS["xmin"], INFOS["xmax"], INFOS["Nx"], endpoint=True)
        y_arr = np.linspace(INFOS["ymin"], INFOS["ymax"], INFOS["Ny"], endpoint=True)
        z_arr = np.linspace(INFOS["zmin"], INFOS["zmax"], INFOS["Nz"], endpoint=True)


        point_idx = [np.argmin(np.abs(INFOS["extract_point"][0]-rx_arr)),
                     np.argmin(np.abs(INFOS["extract_point"][1]-y_arr)),  
                     np.argmin(np.abs(INFOS["extract_point"][2]-z_arr))]  


        log.info("Interpolating E-fields/Gradients and writing to laser file:")
        e_write_shift = int(1)
        b_write_shift = e_write_shift+6*int(INFOS["export_e"])
        egrad_write_shift = b_write_shift+6*int(INFOS["export_b"])
        bgrad_write_shift = egrad_write_shift+18*int(INFOS["export_egrad"])
        no_of_columns = bgrad_write_shift+18*int(INFOS["export_bgrad"])
        # Initialize laser fields file
        laser_file = np.zeros((len(int_t_arr), no_of_columns))  # tsteps, (f_exr, f_eyr, f_ezr or f_bxr, f_byr, f_bzr) #3*2 Exyz (real, imag), #3*2 Bxyz (real, imag), #3*3*2 Grad Exyz (real, imag), #3*3*2 Grad Bxyz (real, imag)
        laser_file[:, 0] = int_t_arr*1E15  # SAVE timesteps in fs
        if INFOS["export_e"] or INFOS["export_egrad"]:

            for fld_count, fld in enumerate(efields):
                fld_arr = np.asarray(h5py.File(INFOS["sim_file_path"], "r")[fld])
                fields_gradients_real = [None]*len(int_t_arr)
                #fields_gradients_imag = [None]*len(int_t_arr)
                for t_count, t_i in enumerate(int_t_arr):
                    fields_gradients_real[t_count] = calc_fields(INFOS, t_arr, rx_arr, y_arr, z_arr, fld, fld_arr, "real", t_i, point_idx, space_tolerance, time_tolerance, int_method)
                    #fields_gradients_imag[t_count] = calc_fields(INFOS, t_arr, rx_arr, y_arr, z_arr, fld, fld_arr, "imag", t_i, point_idx, tolerance, int_method)
                    done = t_count * progress_width // len(int_t_arr)
                    sys.stdout.write("\rProgress for component '%s': [" % (fld) + "=" * done + " " * (progress_width - done) + "] %3i%%" % (done * 100 // progress_width))
                sys.stdout.write("\rProgress for component '%s': ["  % (fld) + "=" * progress_width + " " * (0) + "] %3i%% \n" % (100))
                fields_gradients_real = np.asarray(fields_gradients_real)
                #fields_gradients_imag = np.asarray(fields_gradients_imag)  
                if INFOS["export_e"]:
                    laser_file[:, e_write_shift+fld_count*2] = fields_gradients_real[:, 0]/efield_au_to_v_per_m
                    #laser_file[:, e_write_shift+1+fld_count*2] = fields_gradients_imag[:, 0]/efield_au_to_v_per_m
                if INFOS["export_egrad"]:
                    laser_file[:, egrad_write_shift+fld_count*6], laser_file[:, egrad_write_shift+2+fld_count*6], laser_file[:, egrad_write_shift+4+fld_count*6] =  (fields_gradients_real[:, 1:]/efield_grad_au_to_v_per_m2).T 
                    #laser_file[:, egrad_write_shift+1+fld_count*6], laser_file[:, egrad_write_shift+3+fld_count*6], laser_file[:, egrad_write_shift+5+fld_count*6] =  (fields_gradients_imag[:, 1:]/efield_grad_au_to_v_per_m2).T 
            log.info("E-field/E-gradients extracted!")
        if INFOS["export_b"] or INFOS["export_bgrad"]:
            log.info("Interpolating B-fields/Gradients and writing to laser file:") 
            for fld_count, fld in enumerate(bfields):
                fld_arr = np.asarray(h5py.File(INFOS["sim_file_path"], "r")[fld])
                fields_gradients_real = [None] * len(int_t_arr)
                fields_gradients_imag = [None] * len(int_t_arr)
                for t_count, t_i in enumerate(int_t_arr):
                    fields_gradients_real[t_count] = calc_fields(INFOS, t_arr, rx_arr, y_arr, z_arr, fld, fld_arr, "real", t_i, point_idx, space_tolerance, time_tolerance, int_method)
                    fields_gradients_imag[t_count] = calc_fields(INFOS, t_arr, rx_arr, y_arr, z_arr, fld , fld_arr, "imag", t_i, point_idx, space_tolerance, time_tolerance, int_method)
                    done = t_count * progress_width // len(int_t_arr)
                    sys.stdout.write("\rProgress for component '%s': [" % (fld) + "=" * done + " " * (progress_width - done) + "] %3i%%" % (done * 100 // progress_width))
                sys.stdout.write("\rProgress for component '%s': ["  % (fld) + "=" * progress_width + " " * (0) + "] %3i%% \n" % (100))
                fields_gradients_real = np.asarray(fields_gradients_real)
                fields_gradients_imag = np.asarray(fields_gradients_imag) 
                if INFOS["export_b"]:
                    laser_file[:, b_write_shift+fld_count*2] = fields_gradients_real[:, 0]/bfield_au_to_t 
                    laser_file[:, b_write_shift+1+fld_count*2] = fields_gradients_imag[:, 0]/bfield_au_to_t
                if INFOS["export_bgrad"]:
                    laser_file[:, bgrad_write_shift+fld_count*6], laser_file[:, bgrad_write_shift+2+fld_count*6], laser_file[:, bgrad_write_shift+4+fld_count*6] =  (fields_gradients_real[:, 1:]/bfield_grad_au_to_t_per_m).T 
                    laser_file[:, bgrad_write_shift+1+fld_count*6], laser_file[:, bgrad_write_shift+3+fld_count*6], laser_file[:, bgrad_write_shift+5+fld_count*6] =  (fields_gradients_imag[:, 1:]/bfield_grad_au_to_t_per_m).T 
            log.info("B-field/B-gradients extracted!")
        
        # Obtain frequencies from laser simulation
        freq_arr = extract_frequencies(laser_file, INFOS["export_e"], INFOS["export_b"], INFOS["export_egrad"], INFOS["export_bgrad"], int_t_arr)
        freq_arr = np.asarray(freq_arr)

        head_line_length = 16*(no_of_columns + freq_arr.shape[1]) - 2
        laser_file = np.hstack((laser_file, freq_arr))
        # SAVE LASER FILE
        # header = "t/fs , Re[Erho/x] (au), Im[Erho/x] (au), Re[Ephi/y] (au), Im[Ephi/y] (au), Re[Ez] (au), Im[Ez] (au), \
        # Re[Brho/x] (au), Im[Brho/x] (au), Re[Bphi/y] (au), Im[Brho/y] (au), Re[Bz] (au), Im[Bz] (au)"
        header = f'''\
         laser file 
         SHARC {sharcversion}
         file_version 2.0 
         nsteps = {len(int_t_arr)} 
         dt {INFOS["electronic time_step"]:.8E}
         e-field {str(INFOS["export_e"]).lower()} 
         b-field {str(INFOS["export_b"]).lower()}  
         e-field_gradients {str(INFOS["export_egrad"]).lower()}   
         b-field_gradents {str(INFOS["export_bgrad"]).lower()}    
         laser_freq_path laser_freq'''
        header_line = f''' #{"="*head_line_length}'''+"\n"
        field_columns = ["Time"]
        [field_columns.append(val) for val in ["Re(Ex)", "Im(Ex)", "Re(Ey)", "Im(Ey)", "Re(Ez)", "Im(Ez)"] if INFOS["export_e"]]
        [field_columns.append(val) for val in ["Re(Bx)", "Im(Bx)", "Re(By)", "Im(By)", "Re(Bz)", "Im(Bz)"] if INFOS["export_b"]]
        [field_columns.append(val) for val in ["Re(Ex_grad_x)", "Im(Ex_grad_x)", "Re(Ex_grad_y)", "Im(Ex_grad_y)", "Re(Ex_grad_z)", "Im(Ex_grad_z)", 
                                               "Re(Ey_grad_x)", "Im(Ey_grad_x)", "Re(Ey_grad_y)", "Im(Ey_grad_y)", "Re(Ey_grad_z)", "Im(Ey_grad_z)",
                                               "Re(Ez_grad_x)", "Im(Ez_grad_x)", "Re(Ez_grad_y)", "Im(Ez_grad_y)", "Re(Ez_grad_z)", "Im(Ez_grad_z)"]
            if INFOS["export_egrad"]]
        [field_columns.append(val) for val in ["Re(Bx_grad_x)", "Im(Bx_grad_x)", "Re(Bx_grad_y)", "Im(Bx_grad_y)", "Re(Bx_grad_z)", "Im(Bx_grad_z)",
                                               "Re(By_grad_x)", "Im(By_grad_x)", "Re(By_grad_y)", "Im(By_grad_y)", "Re(By_grad_z)", "Im(By_grad_z)",
                                               "Re(Bz_grad_x)", "Im(Bz_grad_x)", "Re(Bz_grad_y)", "Im(Bz_grad_y)", "Re(Bz_grad_z)", "Im(Bz_grad_z)"]
            if INFOS["export_bgrad"]]
        [field_columns.append(val) for val in ["Freq."]*freq_arr.shape[1]]

        unit_columns = ["[fs]"]+["a.u."]*(len(field_columns)-1)
        # {"[fs] |":>14} {"[a.u.] |":>17} {"[a.u.] |":>17} {"[a.u.] |":>17} {"[a.u.] |":>17} {"[a.u.] |":>17} {"[a.u.] |":>17} 
        #{"[a.u.] |":>17} {"[a.u.] |":>17} {"[a.u.] |":>17} {"[a.u.] |":>17} {"[a.u.] |":>17} {"[a.u.] |":>17} 
        max_lengths = [11]+[13 for column in field_columns[:-1]]
        header = '\n'.join(" ! " + line.lstrip() for line in header.split('\n'))+"\n"
        header_fields = " # " + " | ".join([f"{column:>{length}}" for column, length in zip(field_columns, max_lengths)])+" |"+"\n"
        header_units = " # " + " | ".join([f"{column:>{length}}" for column, length in zip(unit_columns, max_lengths)])+" |"+"\n"
        header=header+header_line+header_fields+header_units+header_line
        log.info("Writing fields and gradients to file:")
        formatted_laser_file = np.array([[custom_formatter(val) for val in row] for row in laser_file], dtype=str)
        np.savetxt("laser", formatted_laser_file, fmt="%s", delimiter="", header=header, comments='')
    
        
    #QA: Where should the laser file be saved?
    # log.info('\n' + f"{'Full input':#^60}" + '\n')
    # for item in INFOS:
    #     log.info(f"{item:<25} {INFOS[item]}")
    # log.info('')
    # setup = question('Do you want to setup the specified calculations?', bool, True)
    # log.info('')

    # if setup:
    #     setup_all(INFOS, t_arr, rx_arr, y_arr, z_arr, chosen_interface)

    close_keystrokes()


# ======================================================================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log.info('\nCtrl+C makes me a sad SHARC ;-(\n')
        quit(0)
