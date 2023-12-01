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
from scipy.interpolate import RegularGridInterpolator
import shutil

from logger import log
from utils import question                                 
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
stepsize = 0.5  # Length of the nuclear dynamics time steps in fs: QA -> take from SHARC
nsubsteps = 25  # Number of substeps for the integration of the electronic EOM: QA -> take from SHARC
efield_au_to_v_per_m = const.physical_constants["Hartree energy"][0]/const.e/const.physical_constants["Bohr radius"][0]
bfield_au_to_t = const.electron_mass*const.physical_constants["Hartree energy"][0]/(const.e*const.physical_constants["reduced Planck constant"][0])
efield_grad_au_to_v_per_m2 = efield_au_to_v_per_m*const.physical_constants["Bohr radius"][0]     
bfield_grad_au_to_t_per_m =  bfield_au_to_t*const.physical_constants["Bohr radius"][0]    


int_method = "cubic"                 
tolerance = 2  # only one works for now

progress_width = 50

sim_file_attrs = ["dimensions", "tmax_si", "rxmin_si_output", "rxmax_si_output", "ymin_si_output", "ymax_si_output", "zmin_si_output", "zmax_si_output"]


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


def check_laser_file_version(string):
    """
    Checks version of laser fields file to attain compatibility with old laser field files without extraction of field gradients
    Requested laser fields file header: "Laser fields file, version 1.0"
    Args:
        string (): 

    Returns:
       True/False (True, if Gradients are provided) 
    """
    if 'Laser fields file' not in string.lower():
        return False
    f = string.split()
    for i, field in enumerate(f):
        if 'version' in field.lower():
            try:
                v = float(f[i + 1])
                if v not in versionneeded:
                    return False
            except IndexError:
                return False
    return True


start_time = time.time()  


def custom_formatter(val: float):
    """
    Formats the laser fields files' values in defined scientific notation
    Args:
        x (int): 

    Returns:
       Formatted laser fields files' values 
    """
    assert isinstance(val, float), "val must be a float!"
    if np.log(val)<=-99:
        val=0.0
    elif np.isnan(val):
        return f'NaN'
    val_form = '{:.6e}'.format(val)  # Format with 3 digits for the exponent
    mantissa, exponent = val_form.split('e')
    sign = '+' if float(mantissa) >= 0 else ''  # Check if positive
    return f'{sign}{mantissa}E{exponent[0]}{exponent[1:].zfill(3)}'


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
    KEYSTROKES = open('KEYSTROKES.tmp', 'w')


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move('KEYSTROKES.tmp', 'KEYSTROKES.extract_laser_fields')


def get_general(INFOS):
    '''This routine questions from the user some general information:
    - FDTD simulatioin output file
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
            try:
                sim_file = h5py.File(sim_file_path, 'r')
            except IOError:
                log.info('Could not open: {sim_file_path}')
            break
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
    # INFOS[''] = 
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
    log.info(f'Script will extract fields at {extract_point} \u03bcm.\n')
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
    INFOS["nuc_dyn_stepsize"] = stepsize*temp_unit_fac
    INFOS["electronic time_step"] = INFOS["nuc_dyn_stepsize"]/no_el_time_step[0] 
    while True:
        log.info('\nPlease enter the desired spatial interpolation step (in nm). Default: 10')
        delta = question('dx/dy/dz:', float, [10])
        if len(no_el_time_step) != 1:
            log.info('Enter one time step!')
            continue
        break
    INFOS["delta"] = delta[0]*1E-9
    return INFOS


# def calc_fields(t_i, point, point_idx, delta, quant, cmplx, method, tol, dim):
def calc_fields(INFOS, t_arr, rx_arr, y_arr, z_arr, quant: str, cmplx: str, readout_time: float, point_idx: list, tol: int, int_method: str):
    assert isinstance(readout_time, float), "readout_time must be a float!"
    assert isinstance(point_idx, list), "point_ipoint_idx must be a list!" 
    assert isinstance(tol, int), "tol must be an integer!"
    assert isinstance(quant, str), "quant must be a string!"
    # QA: Should I couple the tolerance directly to the tolerance or give an error if cubic is expected and tol=1?
    assert isinstance(int_method, str), "int_method must be a string!"
    assert isinstance(cmplx, str), "cmplx must be a string!"

    sim_file = h5py.File(INFOS["sim_file_path"], "r")
    

    if INFOS["dimensions"]=="CARTESIAN": 
        point_idt = np.argmin(np.abs(readout_time-t_arr))
        grid_idx_x = (point_idx[0]-tol, point_idx[0]+tol+1)
        grid_idx_y = (point_idx[1]-tol, point_idx[1]+tol+1)
        grid_idx_z = (point_idx[2]-tol, point_idx[2]+tol+1)
        dx_basis, dy_basis, dz_basis = [np.eye(3)[idx, :]*INFOS["delta"] for idx in range(3)]  
        if (point_idt-tol)<=0 or (point_idt+tol)>=len(t_arr):
            grid = (rx_arr[grid_idx_x[0]:grid_idx_x[1]],
                    y_arr[grid_idx_y[0]:grid_idx_y[1]],
                    z_arr[grid_idx_z[0]:grid_idx_z[1]])
            interpol_point = INFOS["extract_point"]
            if cmplx=="real":
                interp = RegularGridInterpolator(grid, np.real(sim_file[quant][
                                point_idt,
                                point_idx[0]-tol:point_idx[0]+tol+1,  # No fancy indexing allowed!                                                                                                                                                                                                                              
                                point_idx[1]-tol:point_idx[1]+tol+1,                                                                                                                                                                                                                              
                                point_idx[2]-tol:point_idx[2]+tol+1]),
                                                 method=int_method)
            elif cmplx=="imag":
                interp = RegularGridInterpolator(grid, np.imag(sim_file[quant][
                                point_idt, 
                                point_idx[0]-tol:point_idx[0]+tol+1,
                                point_idx[1]-tol:point_idx[1]+tol+1,
                                point_idx[2]-tol:point_idx[2]+tol+1]),
                                                 method=int_method)
        else:
            red_t_arr_idx = (point_idt-tol, point_idt+tol+1) 
            red_t_arr = t_arr[red_t_arr_idx[0]:red_t_arr_idx[1]]
            grid = (red_t_arr,
                    rx_arr[grid_idx_x[0]:grid_idx_x[1]],
                    y_arr[grid_idx_y[0]:grid_idx_y[1]],
                    z_arr[grid_idx_z[0]:grid_idx_z[1]]) 
            interpol_point=[readout_time, *INFOS["extract_point"]]
            
            dx_basis , dy_basis, dz_basis = np.array([[0, *dx_basis], [0, *dy_basis], [0, *dz_basis]])
            if cmplx=="real":
                interp = RegularGridInterpolator(grid, np.real(sim_file[quant][
                                     point_idt-tol:point_idt+tol+1,
                                     point_idx[0]-tol:point_idx[0]+tol+1,  # No fancy indexing allowed!                                                                                                                                                                                                                              
                                     point_idx[1]-tol:point_idx[1]+tol+1,                                                                                                                                                                                                                              
                                     point_idx[2]-tol:point_idx[2]+tol+1]),
                                                 method=int_method)
            elif cmplx=="imag":
                interp = RegularGridInterpolator(grid, np.imag(sim_file[quant][
                                     point_idt-tol:point_idt+tol+1, 
                                     point_idx[0]-tol:point_idx[0]+tol+1,
                                     point_idx[1]-tol:point_idx[1]+tol+1,
                                     point_idx[2]-tol:point_idx[2]+tol+1]),
                                                 method=int_method) 
        fields = interp(interpol_point)[0]
        gradients = [(interp((interpol_point+di_basis))[0]- interp((interpol_point-di_basis))[0])/(2*INFOS["delta"]) for di_basis in [dx_basis, dy_basis, dz_basis]]
        return fields, *gradients 
    else:
        log.info(f'Dimension not implemented yet: {INFOS["dimensions"]}')
        raise IOError
        # QA: Does one have to return a value, if raise IOError?
        return 0


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

        # Initialize laser fields file
        laser_file = np.nan*np.ones((len(int_t_arr), 50))  # tsteps, #3*2 Exyz (real, imag), #3*2 Bxyz (real, imag), #3*3*2 Grad Exyz (real, imag), #3*3*2 Grad Bxyz (real, imag)
        laser_file[:, 0] = int_t_arr*1E15  # SAVE timesteps in fs

        point_idx = [np.argmin(np.abs(INFOS["extract_point"][0]-rx_arr)),
                     np.argmin(np.abs(INFOS["extract_point"][1]-y_arr)),  
                     np.argmin(np.abs(INFOS["extract_point"][2]-z_arr))]  

        efields = ["e_x_data_si", "e_y_data_si", "e_z_data_si"]
        bfields = ["b_x_data_si", "b_y_data_si", "b_z_data_si"]

        log.info("Interpolating E-fields/Gradients and writing to laser file:")
        for fld_count, fld in enumerate(efields):
            fields_gradients_real = []
            fields_gradients_imag = []
            for t_count, t_i in enumerate(int_t_arr):
                # Calculate real fields
                fields_gradients_real.append(calc_fields(INFOS, t_arr, rx_arr, y_arr, z_arr, fld, "real", t_i, point_idx, tolerance, int_method))
                # Calculate imaginary fields
                fields_gradients_imag.append(calc_fields(INFOS, t_arr, rx_arr, y_arr, z_arr, fld, "imag", t_i, point_idx, tolerance, int_method))
                done = t_count * progress_width // len(int_t_arr)
                sys.stdout.write("\rProgress for component '%s': [" % (fld) + "=" * done + " " * (progress_width - done) + "] %3i%%" % (done * 100 // progress_width))
            sys.stdout.write("\rProgress for component '%s': ["  % (fld) + "=" * progress_width + " " * (0) + "] %3i%% \n" % (100))
            fields_gradients_real = np.asarray(fields_gradients_real)
            fields_gradients_imag = np.asarray(fields_gradients_imag)  
            laser_file[:, 1+fld_count*2] = fields_gradients_real[:, 0]/efield_au_to_v_per_m
            laser_file[:, 14+fld_count*6], laser_file[:, 16+fld_count*6], laser_file[:, 18+fld_count*6] =  (fields_gradients_real[:, 1:]/efield_grad_au_to_v_per_m2).T 
            laser_file[:, 2+fld_count*2] = fields_gradients_imag[:, 0]/efield_au_to_v_per_m
            laser_file[:, 15+fld_count*6], laser_file[:, 17+fld_count*6], laser_file[:, 19+fld_count*6] =  (fields_gradients_imag[:, 1:]/efield_grad_au_to_v_per_m2).T 

        log.info("E-field extracted!")
        log.info("Interpolating B-fields/Gradients and writing to laser file:") 
        for fld_count, fld in enumerate(bfields):
            fields_gradients_real = []
            fields_gradients_imag = []
            for t_count, t_i in enumerate(int_t_arr):
                fields_gradients_real.append(calc_fields(INFOS, t_arr, rx_arr, y_arr, z_arr, fld, "real", t_i, point_idx, tolerance, int_method))
                fields_gradients_imag.append(calc_fields(INFOS, t_arr, rx_arr, y_arr, z_arr, fld, "imag", t_i, point_idx, tolerance, int_method))
                done = t_count * progress_width // len(int_t_arr)
                sys.stdout.write("\rProgress for component '%s': [" % (fld) + "=" * done + " " * (progress_width - done) + "] %3i%%" % (done * 100 // progress_width))
            sys.stdout.write("\rProgress for component '%s': ["  % (fld) + "=" * progress_width + " " * (0) + "] %3i%% \n" % (100))
            fields_gradients_real = np.asarray(fields_gradients_real)
            fields_gradients_imag = np.asarray(fields_gradients_imag) 
            laser_file[:, 7+fld_count*2] = fields_gradients_real[:, 0]/bfield_au_to_t 
            laser_file[:, 32+fld_count*6], laser_file[:, 34+fld_count*6], laser_file[:, 36+fld_count*6] =  (fields_gradients_real[:, 1:]/bfield_grad_au_to_t_per_m).T 
            laser_file[:, 8+fld_count*2] = fields_gradients_imag[:, 0]/bfield_au_to_t
            laser_file[:, 33+fld_count*6], laser_file[:, 35+fld_count*6], laser_file[:, 37+fld_count*6] =  (fields_gradients_imag[:, 1:]/bfield_grad_au_to_t_per_m).T 
        log.info("B-field extracted!")
        # SAVE LASER FILE
        # header = "t/fs , Re[Erho/x] (au), Im[Erho/x] (au), Re[Ephi/y] (au), Im[Ephi/y] (au), Re[Ez] (au), Im[Ez] (au), \
        # Re[Brho/x] (au), Im[Brho/x] (au), Re[Bphi/y] (au), Im[Brho/y] (au), Re[Bz] (au), Im[Bz] (au)"
        header = f'''\
        ! Laser file SHARC {sharcversion}
        ! version 1.0 
        ! nsteps = {len(int_t_arr)} 
        ! dt =  {INFOS["electronic time_step"]}
        ! E-fields = true 
        ! B-fields = true
        ! E-field gradients = true 
        ! B-field gradients = true 
        '''
        header = '\n'.join(line.lstrip() for line in header.split('\n'))
        log.info("Writing fields and gradients to file:")
        formatted_laser_file = np.array([[custom_formatter(val) for val in row] for row in laser_file], dtype=str)
        np.savetxt("laser", formatted_laser_file, fmt="%s", delimiter="  ", header=header, comments='')
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
