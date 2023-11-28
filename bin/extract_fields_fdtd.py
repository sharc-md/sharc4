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

version = '1.0'                                                                                                                                
versionneeded = [1.0, float(version)]                                                                                           
versiondate = datetime.date(2023, 8, 24)                                                                                                       
global KEYSTROKES                                                                                                                              
old_question = question                                                                                                                        

# UNIT FACTORS
spat_unit_fac = 1E-6  # Conversion input unit to SI
temp_unit_fac = 1E-15  # Conversion input unit to SI
efield_au_to_v_per_m = const.physical_constants["Hartree energy"][0]/const.e/const.physical_constants["Bohr radius"][0]
bfield_au_to_t = const.electron_mass*const.physical_constants["Hartree energy"][0]/(const.e*const.physical_constants["reduced Planck constant"][0])
efield_grad_au_to_v_per_m2 = efield_au_to_v_per_m*const.physical_constants["Bohr radius"][0]     
bfield_grad_au_to_t_per_m =  bfield_au_to_t*const.physical_constants["Bohr radius"][0]    


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
    val_form = '{:.3e}'.format(val)  # Format with 3 digits for the exponent
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
        sim_file = 'sim_file.hdf5'
        if os.path.exists(sim_file):
            log.info('FDTD simulation output file "sim_file.hdf5" detected. Do you want to use this?')
            if not question('Use file "sim_file.hdf5"?', bool, True):
                raise IOError
        else:
            raise IOError
    except IOError:
        log.info('\nIf you do not have an initial conditions file, prepare one with MEEP!\n')
        log.info('Please enter the filename of the FDTD simulation output file.')
        while True:
            sim_file = question('FDTD simulation output filename:', str, 'FDTD output')
            sim_file = os.path.expanduser(os.path.expandvars(sim_file))
            if os.path.isdir(sim_file):
                log.info('Is a directory: %s' % (sim_file))
                continue
            if not os.path.isfile(sim_file):
                log.info('File does not exist: %s' % (sim_file))
                continue
            try:
                simf = h5py.File(sim_file, 'r')
            except IOError:
                log.info('Could not open: %s' % (sim_file))
            # QA: What should be additional tests for the read-in of an FDTD file?
            #   -> existance of xmin, xmax, ... 
            #   -> existance of dimensions (CARTESIAN/CYLINDRICAL) in sim_file
            #     continue
            # line = simf.readline()
            # if check_initcond_version(line):
            #     break
            # else:
            #     log.info('File does not contain initial conditions!')
            #     continue
    # read the header
    INFOS["dimensions"] = sim_file.attrs["dimensions"]
    if sim_file.attrs["dimensions"]=="CARTESIAN":
        INFOS["tmin"], INFOS["xmin"], INFOS["ymin"], INFOS["zmin"] = [0] + [sim_file.attrs[var] for var in ["rxmin_si_output", "ymin_si_output", "zmin_si_output"]]
        INFOS["tmax"], INFOS["xmax"], INFOS["ymax"], INFOS["zmax"] = [sim_file.attrs[var] for var in ["tmax_si", "rxmax_si_output", "ymax_si_output", "zmax_si_output"]] 
        # tmin, tmax, tres = [0, sim_file.attrs["tmax_si"], readout_file.attrs["saved_dt_si"]]
        # rxres, zres = [sim_file.attrs["drx_si_output"], readout_file.attrs["dz_si_output"]]
        INFOS["Nt"], INFOS["Nrx"], INFOS["Ny"], INFOS["Nz"] = sim_file["e_x_data_si"].shape

    else:
        # INFOS['tmin'], INFOS['xmin'], INFOS['zmin'] = [0] + [sim_file.attrs[var] for var in ["rxmin_si_output", "zmin_si_output"]]
        # INFOS['tmax'], INFOS['xmax'], INFOS['zmax'] = [sim_file.attrs[var] for var in ["tmax_si", "rxmax_si_output", "zmax_si_output"]] 
        # INFOS['Nt'], INFOS['Nrx'], INFOS['Nz'] = sim_file['e_x_data_si'].shape
        log.info('Cylindrical coordinates not implemented yet!')
        raise IOError
    INFOS['simf'] = simf
    log.info('\nFile "%s" contains simulation output in %s coordinates.' % (sim_file, sim_file.attrs['dimensions']))
    log.info("Fields are saved for the following coordinates:")
    log.info(f'r/x (µm): ({INFOS["xmin"]/spat_unit_fac:.2f}, {INFOS["xmax"]*1E6:.2f})')
    if sim_file.attrs['dimensions']=="CARTESIAN":
        log.info(f'y (µm): ({INFOS["ymin"]/spat_unit_fac:.2f}, {INFOS["ymax"]/spat_unit_fac:.2f})')
    log.info(f'z (µm): ({INFOS["zmin"]/spat_unit_fac:.2f}, {INFOS["zmax"]/spat_unit_fac:.2f})')
    log.info("------------------------------")
    # QA: logging module probably has some problems with f-strings 
    #   -> Should I switch to other formatting?
    # QA: Which units should be default in input?
    #   -> Would suggest µm, Angstrom
    log.info('\nPlease enter the laser field extraction positions (in µm) as three floats separated by space. Default: [0, 0, 0]')
    while True:
        extract_point = question('Extraction point:', float, [0., 0., 0.])  # Default extraction point at equilibrium
        if len(extract_point) != 3:
            log.info('Enter three numbers separated by spaces!')
            continue
        if (INFOS["xmin"] > extract_point[0]*spat_unit_fac) or (INFOS["xmax"] < extract_point[0]*spat_unit_fac):
            log.info(f'X-coordinate of extraction point {extract_point[0]} must lie within {INFOS["xmin"]/spat_unit_fac:.2f} \u03bcm, {INFOS["xmax"]/spat_unit_fac:.2f} \u03bcm ]!')
            continue
        if (INFOS["ymin"] > extract_point[1]*spat_unit_fac) or (INFOS["ymax"] < extract_point[1]*spat_unit_fac):
            log.info(f'Y-coordinate of extraction point {extract_point[1]} must lie within {INFOS["ymin"]/spat_unit_fac:.2f} \u03bcm, {INFOS["ymax"]/spat_unit_fac:.2f} \u03bcm ]!')
            continue
        if (INFOS["zmin"] > extract_point[2]*spat_unit_fac) or (INFOS["zmax"] < extract_point[2]*spat_unit_fac):
            log.info(f'Z-coordinate of extraction point {extract_point[2]} must lie within {INFOS["zmin"]/spat_unit_fac:.2f} \u03bcm, {INFOS["zmax"]/spat_unit_fac:.2f} \u03bcm ]!')
            continue
        break
    log.info(f'\nScript will extract fields at {extract_point} / \u03bcm.\n')
    INFOS["extract_point"] = extract_point*spat_unit_fac
    # QA: time step / resolution
    #   -> what should be the default, default unit
    log.info('\nPlease enter the time step size (in fs). Default: 1.0 fs')
    while True:
        time_step = question('Time step:', float, 1.0)  # Default time step 
        if len(extract_point) != 1:
            log.info('Enter one time step!')
            continue
        break
    INFOS["time_step"] = time_step*temp_unit_fac 
    return INFOS


int_t_arr = np.linspace(tmin, tmax, 3*Nt) # HARD CODED interpolation time array

# Initialize laser fields file
laser_file = 50*np.ones((len(int_t_arr), 50))  # tsteps, #3*2 Exyz (real, imag), #3*2 Bxyz (real, imag), #3*3*2 Grad Exyz (real, imag), #3*3*2 Grad Bxyz (real, imag)
laser_file[:, 0] = int_t_arr*1E15  # SAVE timesteps in fs



# def calc_fields(t_i, point, point_idx, delta, quant, cmplx, method, tol, dim):
def calc_fields(INFOS, quant: str, cmplx: str, readout_time: float, point_idx: list, tol: int, int_method: str, dx: float):
    assert isinstance(readout_time, float), "readout_time must be a float!"
    assert isinstance(point_idx, list), "point_ipoint_idx must be a list!" 
    assert isinstance(tol, int), "tol must be an integer!"
    assert isinstance(dx, float), "dx must be a float!"
    assert isinstance(quant, str), "quant must be a string!"
    # QA: Should I couple the tolerance directly to the tolerance or give an error if cubic is expected and tol=1?
    assert isinstance(int_method, str), "int_method must be a string!"
    assert isinstance(cmplx, str), "cmplx must be a string!"

    t_arr = np.linspace(INFOS["tmin"], INFOS["tmax"], INFOS["Nt"], endpoint=True)
    rx_arr = np.linspace(INFOS["xmin"], INFOS["xmax"], INFOS["Nx"], endpoint=True)
    y_arr = np.linspace(INFOS["ymin"], INFOS["ymax"], INFOS["Ny"], endpoint=True)
    z_arr = np.linspace(INFOS["zmin"], INFOS["zmax"], INFOS["Nz"], endpoint=True)
    sim_file = INFOS["sim_file"]

    if INFOS["dimensions"]=="CARTESIAN": 
        point_idt = np.argmin(np.abs(readout_time-t_arr))
        grid_idx_x = (point_idx[0]-tol, point_idx[0]+tol+1)
        grid_idx_y = (point_idx[1]-tol, point_idx[1]+tol+1)
        grid_idx_z = (point_idx[2]-tol, point_idx[2]+tol+1)
        dx_basis, dy_basis, dz_basis = [np.eye(3)[idx, :]*dx for idx in range(3)]  
        if (point_idt-tol)<=0 or (point_idt+tol)>=len(t_arr):
            grid = (rx_arr[grid_idx_x[0]:grid_idx_x[1]]*1E6,
                    y_arr[grid_idx_y[0]:grid_idx_y[1]]*1E6,
                    z_arr[grid_idx_z[0]:grid_idx_z[1]]*1E6)
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
                    rx_arr[grid_idx_x[0]:grid_idx_x[1]]*1E6,
                    y_arr[grid_idx_y[0]:grid_idx_y[1]]*1E6,
                    z_arr[grid_idx_z[0]:grid_idx_z[1]]*1E6) 
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
        gradients = [(interp((interpol_point+di_basis))[0]- interp((interpol_point-di_basis))[0])/(2*dx) for di_basis in [dx_basis, dy_basis, dz_basis]]
        return fields, *gradients 
    else:
        log.info(f'Dimension not implemented yet: {INFOS["dimensions"]}')
        raise IOError
        # QA: Does one have to return a value, if raise IOError?
        return 0



efields = ["e_x_data_si", "e_y_data_si", "e_z_data_si"]
bfields = ["b_x_data_si", "b_y_data_si", "b_z_data_si"]
for fld_count, fld in enumerate(efields):
    fields_gradients = np.asarray([field_grid_point(t_i, readout_point, point_idx, delta, fld, "real", int_method, tolerance, dimensions) for t_i in int_t_arr])
    laser_file[:, 1+fld_count*2] = fields_gradients[:, 0]/efield_au_to_v_per_m
    laser_file[:, 14+fld_count*6], laser_file[:, 16+fld_count*6], laser_file[:, 18+fld_count*6] =  (fields_gradients[:, 1:]/efield_grad_au_to_v_per_m2).T 
    fields_gradients = np.asarray([field_grid_point(t_i, readout_point, point_idx, delta, fld, "imag", int_method, tolerance, dimensions) for t_i in int_t_arr])
    laser_file[:, 2+fld_count*2] = fields_gradients[:, 0]/efield_au_to_v_per_m
    laser_file[:, 15+fld_count*6], laser_file[:, 17+fld_count*6], laser_file[:, 19+fld_count*6] =  (fields_gradients[:, 1:]/efield_grad_au_to_v_per_m2).T 
print("E-field extracted!")
for fld_count, fld in enumerate(bfields):
    fields_gradients = np.asarray([field_grid_point(t_i, readout_point, point_idx, delta, fld, "real", int_method, tolerance, dimensions) for t_i in int_t_arr])
    laser_file[:, 7+fld_count*2] = fields_gradients[:, 0]/bfield_au_to_t
    laser_file[:, 32+fld_count*6], laser_file[:, 34+fld_count*6], laser_file[:, 36+fld_count*6] =  (fields_gradients[:, 1:]/bfield_grad_au_to_t_per_m).T 
    fields_gradients = np.asarray([field_grid_point(t_i, readout_point, point_idx, delta, fld, "imag", int_method, tolerance, dimensions) for t_i in int_t_arr])
    laser_file[:, 8+fld_count*2] = fields_gradients[:, 0]/bfield_au_to_t
    laser_file[:, 33+fld_count*6], laser_file[:, 35+fld_count*6], laser_file[:, 37+fld_count*6] =  (fields_gradients[:, 1:]/bfield_grad_au_to_t_per_m).T 
print("B-field extracted!")


print(f"Laser fields will be taken @ {np.asarray(readout_point)} (µm)")
# WRITE LASER FILE: https://sharc-md.org/?page_id=50#tth_sEc4.5



#SAVE LASER FILE
#header = ["t/fs" , "Ex_r/au" "Ex_i/au", "Ey_r/au", "Ey_i/au", "Ez_r/au", "Ez_i/au", "Bx_r/au" "Bx_i/au", "By_r/au", "By_i/au", "Bz_r/au", "Bz_i/au"]
header = "t/fs , Re[Erho/x] (au), Im[Erho/x] (au), Re[Ephi/y] (au), Im[Ephi/y] (au), Re[Ez] (au), Im[Ez] (au), \
Re[Brho/x] (au), Im[Brho/x] (au), Re[Bphi/y] (au), Im[Brho/y] (au), Re[Bz] (au), Im[Bz] (au)"

formatted_laser_file = np.array([[custom_formatter(val) for val in row] for row in laser_file], dtype=str)
np.savetxt("laser_"+str(readout_point[0]), formatted_laser_file, fmt="%s", delimiter="  ", header=header)
#np.savetxt("laser", laser_file, delimiter="  ", header=header)


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
    # chosen_interface: SHARC_INTERFACE = get_interface()()
    INFOS = get_requests(INFOS, chosen_interface)
    INFOS = chosen_interface.get_infos(INFOS, KEYSTROKES)
    INFOS = get_runscript_info(INFOS)

    log.info('\n' + f"{'Full input':#^60}" + '\n')
    for item in INFOS:
        log.info(f"{item:<25} {INFOS[item]}")
    log.info('')
    setup = question('Do you want to setup the specified calculations?', bool, True)
    log.info('')

    if setup:
        setup_all(INFOS, chosen_interface)

    close_keystrokes()


# ======================================================================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log.info('\nCtrl+C makes me a sad SHARC ;-(\n')
        quit(0)
